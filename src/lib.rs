use std::fs::File;

use bevy::prelude::*;
use bevy::render::settings::{PowerPreference, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy_egui::{EguiContexts, EguiPlugin};
use bevy_flycam::{FlyCam, MovementSettings, NoCameraPlayerPlugin};
use bevy_sefirot::kernel;
use bevy_sefirot::luisa::{InitKernel, LuisaDevice, LuisaPlugin};
use luisa::lang::types::vector::Vec3 as LVec3;
use luisa_compute::DeviceType;
use sefirot::graph::{AsNodes as AsNodesExt, ComputeGraph};
use sefirot::mapping::buffer::StaticDomain;
use sefirot::prelude::*;
use serde::{Deserialize, Serialize};

mod simulation;
use simulation::*;
pub mod data;
use data::*;

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
        .insert_resource(ClearColor(Color::srgb(0.7, 0.7, 0.72)))
        .insert_resource(MovementSettings {
            sensitivity: 0.00015,
            speed: 30.0,
        })
        .add_plugins(LuisaPlugin {
            device: DeviceType::Cuda,
            ..default()
        })
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
            ),
        )
        .add_systems(Update, (step, update_ui, update_render).chain())
        .run();
}

fn lv(a: Vec3) -> LVec3<f32> {
    LVec3::new(a.x, a.y, a.z)
}

fn setup(
    mut commands: Commands,
    device: Res<LuisaDevice>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let args = std::env::args().collect::<Vec<_>>();
    let scene_name = args.get(1).cloned().unwrap_or("scene.ron".to_string());
    let scene = ron::de::from_reader::<_, data::Scene>(File::open(scene_name).unwrap()).unwrap();
    let scene = scene.load();
    let constants = scene.constants;

    let mesh = meshes.add(Sphere::new(0.5));

    for object in &scene.objects {
        let material = materials.add(StandardMaterial {
            base_color: object.color,
            perceptual_roughness: 1.0,
            ..default()
        });
        let fixed_material = materials.add(StandardMaterial {
            base_color: object.color.mix(&Color::BLACK, 0.5),
            ..default()
        });

        for index in object.particle_start..object.particle_start + object.particle_count {
            let particle = &scene.particles[index as usize];
            commands
                .spawn(PbrBundle {
                    mesh: mesh.clone(),
                    material: if particle.inv_mass == 0.0 {
                        fixed_material.clone()
                    } else {
                        material.clone()
                    },
                    transform: Transform::from_translation(particle.position),
                    ..default()
                })
                .insert(ObjectParticle { index });
        }
    }

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 1000.0,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_rotation_x(-1.0)),
        ..default()
    });

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
        fixed: particles.iter().map(|p| p.inv_mass == 0.0).collect(),
    };

    let particles = simulation::Particles {
        domain: StaticDomain::<1>::new(l as u32),
        position: device.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        predicted_position: device.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        displacement: device.create_buffer_from_fn(l, |i| lv(particles[i].velocity * constants.dt)),
        rest_position: device.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        bond_start: device.create_buffer_from_fn(l, |i| particles[i].bond_start),
        bond_count: device.create_buffer_from_fn(l, |i| particles[i].bond_count),
        inv_mass: device.create_buffer_from_fn(l, |i| particles[i].inv_mass),
    };
    let bonds = Bonds {
        other_particle: device
            .create_buffer_from_fn(scene.bonds.len(), |i| scene.bonds[i].other_particle),
    };
    let grid_size = constants.grid_size.element_product() as usize;
    let grid = Grid {
        domain: StaticDomain::<1>::new(grid_size as u32),
        count: device.create_buffer(grid_size),
        offset: device.create_buffer(grid_size),
        particles: device.create_buffer(l),
        next_block: device.create_buffer(1),
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
}

#[derive(Debug, Resource)]
struct Controls {
    slice: bool,
    slice_position: f32,
    running: bool,
    visualize_bonds: bool,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            slice: false,
            running: false,
            slice_position: 0.01, // mostly to get r-a to shut up.
            visualize_bonds: false,
        }
    }
}

fn update_ui(mut contexts: EguiContexts, mut controls: ResMut<Controls>) {
    egui::Window::new("Controls").show(contexts.ctx_mut(), |ui| {
        ui.checkbox(&mut controls.running, "Running");
        ui.checkbox(&mut controls.slice, "Slice");
        ui.add(
            egui::Slider::new(&mut controls.slice_position, -20.0..=20.0).text("Slice Position"),
        );
        ui.checkbox(&mut controls.visualize_bonds, "Render Bonds");
    });
}

fn update_render(
    mut gizmos: Gizmos,
    controls: Res<Controls>,
    data: Res<ParticleBondData>,
    bonds: Res<Bonds>,
    particles: Res<simulation::Particles>,
    mut query: Query<(&ObjectParticle, &mut Transform, &mut Visibility)>,
) {
    let positions = particles.position.copy_to_vec();
    let bonds = bonds.other_particle.copy_to_vec();

    for (particle, mut transform, mut visible) in query.iter_mut() {
        if controls.slice && transform.translation.x > controls.slice_position {
            *visible = Visibility::Hidden;
        } else {
            *visible = Visibility::Visible;
        }
        // Update nalgebra eventually.
        let pos = positions[particle.index as usize];
        let pos = Vec3::new(pos.x, pos.y, pos.z);
        transform.translation = pos;
        if controls.visualize_bonds && !data.fixed[particle.index as usize] {
            let start = data.bond_start[particle.index as usize] as usize;
            let count = data.bond_count[particle.index as usize] as usize;
            for &bond in bonds.iter().skip(start).take(count) {
                if bond != u32::MAX {
                    let other = bond as usize;
                    let other_pos = positions[other];
                    let other_pos = Vec3::new(other_pos.x, other_pos.y, other_pos.z);
                    gizmos.line(pos, other_pos, Color::WHITE);
                }
            }
        }
    }
}

#[derive(Debug, Resource)]
struct ParticleBondData {
    bond_start: Vec<u32>,
    bond_count: Vec<u32>,
    fixed: Vec<bool>,
}
