use std::fs::File;

use bevy::prelude::*;
use bevy::render::settings::{PowerPreference, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy_egui::{EguiContexts, EguiPlugin};
use bevy_flycam::{FlyCam, MovementSettings, NoCameraPlayerPlugin};
use bevy_sefirot::kernel;
use bevy_sefirot::luisa::{InitKernel, LuisaPlugin};
use luisa::lang::types::vector::{Vec3 as LVec3, Vec4 as LVec4};
use sefirot::graph::{AsNodes as AsNodesExt, ComputeGraph};
use sefirot::mapping::buffer::StaticDomain;
use sefirot::prelude::*;
use serde::{Deserialize, Serialize};

mod simulation;
use simulation::*;
pub mod data;
use data::*;
mod cosserat;
mod utils;

fn install_eyre() {
    use color_eyre::config::*;
    HookBuilder::blank()
        .capture_span_trace_by_default(true)
        .add_frame_filter(Box::new(|frames| {
            let allowed = &["sefirot", "fracture"];
            frames.retain(|frame| {
                allowed.iter().any(|f| {
                    let name = if let Some(name) = frame.name.as_ref() {
                        name.as_str()
                    } else {
                        return false;
                    };

                    name.starts_with(f)
                })
            });
        }))
        .install()
        .unwrap();
}

pub fn main() {
    install_eyre();

    App::new()
        .add_plugins(
            DefaultPlugins.set(RenderPlugin {
                render_creation: WgpuSettings {
                    power_preference: PowerPreference::HighPerformance, // Swap to LowPower for igpu.
                    ..default()
                }
                .into(),
                ..default()
            }),
        )
        .add_plugins(EguiPlugin)
        // Potentially replace with the fancy camera controller.
        .add_plugins(NoCameraPlayerPlugin)
        .init_resource::<Controls>()
        .insert_resource(ClearColor(Color::srgb(0.6, 0.6, 0.62)))
        .insert_resource(MovementSettings {
            sensitivity: 0.00015,
            speed: 30.0,
        })
        .add_plugins(LuisaPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            InitKernel,
            (
                init_count_kernel,
                init_reset_grid_kernel,
                init_add_particle_kernel,
                init_compute_offset_kernel,
                init_predict_kernel,
                init_solve_kernel,
                init_solve_update_kernel,
            ),
        )
        .add_systems(Update, (step, update_ui, update_render).chain())
        .run();
}

fn lv(a: Vec3) -> LVec3<f32> {
    LVec3::new(a.x, a.y, a.z)
}

#[derive(Resource)]
struct Palette {
    materials: Vec<Vec<Handle<StandardMaterial>>>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ambient: ResMut<AmbientLight>,
) {
    let args = std::env::args().collect::<Vec<_>>();
    let scene_name = args.get(1).cloned().unwrap_or("scene.ron".to_string());
    let scene = ron::de::from_reader::<_, data::Scene>(File::open(scene_name).unwrap()).unwrap();
    let scene = scene.load();
    let constants = scene.constants;

    let mesh = meshes.add(Sphere::new(0.5));

    let mut palette = vec![];
    for (i, object) in scene.objects.iter().enumerate() {
        let mut object_palette = vec![];
        let base = Oklcha::from(object.color);
        let transparent = base.alpha < 1.0;
        for i in 0..20 {
            let color = base
                .with_lightness(
                    base.lightness
                        + object.lightness_multiplier * (i as f32).powf(object.lightness_power),
                )
                .with_alpha(base.alpha.lerp(1.0, i as f32 / 19.0));
            let material = materials.add(StandardMaterial {
                base_color: color.into(),
                perceptual_roughness: 1.0,
                alpha_mode: if transparent {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                },
                ..default()
            });
            object_palette.push(material);
        }
        let fixed_material = materials.add(StandardMaterial {
            base_color: base.mix(&Color::BLACK.into(), 0.3).into(),
            perceptual_roughness: 1.0,
            ..default()
        });

        for index in object.particle_start..object.particle_start + object.particle_count {
            let particle = &scene.particles[index as usize];
            commands
                .spawn(PbrBundle {
                    mesh: mesh.clone(),
                    material: if particle.mass == f32::INFINITY {
                        fixed_material.clone()
                    } else {
                        object_palette[0].clone()
                    },
                    transform: Transform::from_translation(particle.position),
                    ..default()
                })
                .insert(ObjectParticle {
                    index,
                    object: i as u32,
                });
        }
        palette.push(object_palette);
    }
    commands.insert_resource(Palette { materials: palette });

    if constants.ambient_only {
        ambient.brightness = 1000.0;
    } else {
        commands.spawn(DirectionalLightBundle {
            directional_light: DirectionalLight {
                color: Color::WHITE,
                illuminance: 1000.0,
                ..default()
            },
            transform: Transform::from_rotation(Quat::from_rotation_x(-1.0)),
            ..default()
        });
    }

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(constants.camera_position)
                .looking_at(constants.camera_target, Vec3::Y),
            projection: PerspectiveProjection {
                fov: 1.15,
                ..default()
            }
            .into(),
            ..default()
        },
        FlyCam,
    ));

    let particles = scene.particles;
    let l = particles.len();
    println!("Num particles: {}", l);

    let render = ParticleBondData {
        bond_start: particles.iter().map(|p| p.bond_start).collect(),
        bond_count: particles.iter().map(|p| p.bond_count).collect(),
        fixed: particles.iter().map(|p| p.mass == f32::INFINITY).collect(),
    };

    let mut this_particle = vec![0u32; scene.bonds.len()];
    for (ix, p) in particles.iter().enumerate() {
        for i in p.bond_start..p.bond_start + p.bond_count {
            this_particle[i as usize] = ix as u32;
        }
    }
    let bonds = Bonds {
        other_particle: DEVICE
            .create_buffer_from_fn(scene.bonds.len(), |i| scene.bonds[i].other_particle),
        rest_rotation: DEVICE.create_buffer_from_fn(scene.bonds.len(), |i| {
            let dir = particles[scene.bonds[i].other_particle as usize].position
                - particles[this_particle[i] as usize].position;
            let dir = dir.normalize();
            let q = Quat::from_rotation_arc(Vec3::Z, dir);
            LVec4::new(q.x, q.y, q.z, q.w)
        }),
        length: DEVICE.create_buffer_from_fn(scene.bonds.len(), |i| {
            (particles[scene.bonds[i].other_particle as usize].position
                - particles[this_particle[i] as usize].position)
                .length()
        }),
    };

    let particles = simulation::Particles {
        domain: StaticDomain::<1>::new(l as u32),
        linpos: DEVICE.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        angpos: DEVICE.create_buffer_from_fn(l, |_i| LVec4::new(0.0, 0.0, 0.0, 1.0)),
        last_linpos: DEVICE.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        last_angpos: DEVICE.create_buffer_from_fn(l, |_i| LVec4::new(0.0, 0.0, 0.0, 1.0)),
        linvel: DEVICE.create_buffer_from_fn(l, |i| lv(particles[i].velocity * constants.dt)),
        angvel: DEVICE.create_buffer_from_fn(l, |_i| LVec3::splat(0.0)),
        last_linvel: DEVICE.create_buffer_from_fn(l, |i| lv(particles[i].velocity * constants.dt)),
        last_angvel: DEVICE.create_buffer_from_fn(l, |_i| LVec3::splat(0.0)),
        bond_start: DEVICE.create_buffer_from_fn(l, |i| particles[i].bond_start),
        bond_count: DEVICE.create_buffer_from_fn(l, |i| particles[i].bond_count),
        broken: matches!(constants.breaking_model, BreakingModel::TotalStress { .. })
            .then(|| DEVICE.create_buffer_from_fn(l, |_i| false)),
        mass: DEVICE.create_buffer_from_fn(l, |i| particles[i].mass),
    };
    let grid_size = constants.grid_size.element_product() as usize;
    let grid = Grid {
        domain: StaticDomain::<1>::new(grid_size as u32),
        count: DEVICE.create_buffer(grid_size),
        offset: DEVICE.create_buffer(grid_size),
        particles: DEVICE.create_buffer(l),
        next_block: DEVICE.create_buffer(1),
    };
    commands.insert_resource(render);
    commands.insert_resource(particles);
    commands.insert_resource(bonds);
    commands.insert_resource(grid);
    commands.insert_resource(constants);
}

#[derive(Debug, Component)]
struct ObjectParticle {
    index: u32,
    object: u32,
}

#[derive(Debug, Resource)]
struct Controls {
    slice: bool,
    slice_position: f32,
    running: bool,
    visualize_bonds: bool,
    remove_singletons: bool,
    lock: bool,
    render_hidden_bonds: bool,
    render_visible_bonds: bool,
    render_absolute: bool,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            slice: false,
            running: false,
            slice_position: 0.0,
            visualize_bonds: false,
            remove_singletons: false,
            lock: false,
            render_hidden_bonds: true,
            render_visible_bonds: true,
            render_absolute: false,
        }
    }
}

fn update_ui(mut contexts: EguiContexts, mut controls: ResMut<Controls>) {
    egui::Window::new("Controls").show(contexts.ctx_mut(), |ui| {
        ui.checkbox(&mut controls.running, "Running");
        ui.checkbox(&mut controls.slice, "Slice");
        ui.add(
            egui::Slider::new(&mut controls.slice_position, -100.0..=100.0).text("Slice Position"),
        );
        ui.checkbox(&mut controls.visualize_bonds, "Render Bonds");
        ui.checkbox(&mut controls.remove_singletons, "Remove Single Particles");
        ui.checkbox(&mut controls.lock, "Lock");
        ui.checkbox(&mut controls.render_hidden_bonds, "Render Hidden Bonds");
        ui.checkbox(&mut controls.render_visible_bonds, "Render Visible Bonds");
        ui.checkbox(&mut controls.render_absolute, "Use Absolute Bond Colors");
    });
}

fn update_render(
    mut gizmos: Gizmos,
    controls: Res<Controls>,
    data: Res<ParticleBondData>,
    bonds: Res<Bonds>,
    particles: Res<simulation::Particles>,
    palette: Res<Palette>,
    mut query: Query<(
        &ObjectParticle,
        &mut Transform,
        &mut Visibility,
        &mut Handle<StandardMaterial>,
    )>,
) {
    let positions = particles.last_linpos.copy_to_vec();
    let bonds = bonds.other_particle.copy_to_vec();
    let broken = particles.broken.as_ref().map(|b| b.copy_to_vec());

    for (particle, mut transform, mut visible, mut material) in query.iter_mut() {
        let lock = controls.lock;

        let pos = positions[particle.index as usize];
        let pos = Vec3::new(pos.x, pos.y, pos.z);
        if !lock {
            if controls.slice && transform.translation.x > controls.slice_position {
                *visible = Visibility::Hidden;
            } else {
                *visible = Visibility::Visible;
            }
        }
        transform.translation = pos;
        let start = data.bond_start[particle.index as usize] as usize;
        let count = data.bond_count[particle.index as usize] as usize;

        let mut num_bonds = 0;

        if !data.fixed[particle.index as usize] {
            let broken = broken
                .as_ref()
                .map(|b| b[particle.index as usize])
                .unwrap_or(false);
            for &bond in bonds.iter().skip(start).take(count) {
                if bond != u32::MAX {
                    num_bonds += 1;
                    if !broken
                        && controls.visualize_bonds
                        && (controls.render_hidden_bonds || *visible == Visibility::Visible)
                        && (controls.render_visible_bonds || *visible == Visibility::Hidden)
                    {
                        let other = bond as usize;
                        let other_pos = positions[other];
                        let other_pos = Vec3::new(other_pos.x, other_pos.y, other_pos.z);
                        gizmos.line(pos, other_pos, Color::WHITE);
                    }
                }
            }
            if broken {
                num_bonds = 0;
            }
            if !lock && num_bonds == 0 && controls.remove_singletons {
                *visible = Visibility::Hidden;
            }
            let material_index = ((1.0
                - num_bonds as f32
                    / if controls.render_absolute {
                        16.0
                    } else {
                        count as f32
                    })
                * 19.99)
                .clamp(0.0, 19.99)
                .floor() as usize;
            *material = palette.materials[particle.object as usize][material_index].clone();
        }
    }
}

#[derive(Debug, Resource)]
struct ParticleBondData {
    bond_start: Vec<u32>,
    bond_count: Vec<u32>,
    fixed: Vec<bool>,
}
