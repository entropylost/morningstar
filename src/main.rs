use std::fs::File;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::settings::{PowerPreference, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy_egui::{EguiContexts, EguiPlugin};
use bevy_flycam::{MovementSettings, PlayerPlugin};
use bevy_sefirot::kernel;
use bevy_sefirot::luisa::{InitKernel, LuisaDevice, LuisaPlugin};
use luisa::lang::types::vector::Vec3 as LVec3;
use luisa_compute::DeviceType;
use parking_lot::Mutex;
use sefirot::graph::{AsNodes as AsNodesExt, ComputeGraph};
use sefirot::mapping::buffer::StaticDomain;
use sefirot::prelude::*;

#[derive(Debug, Value, Clone, Copy)]
#[repr(C)]
struct Particle {
    position: LVec3<f32>,
    next_position: LVec3<f32>,
    velocity: LVec3<f32>,
    next_velocity: LVec3<f32>,
    // Used for computing the rest bond length.
    rest_position: LVec3<f32>,
    bond_start: u32,
    bond_count: u32,
    fixed: bool,
}

#[derive(Debug, Value, Clone, Copy)]
#[repr(C)]
struct Bond {
    // Broken if its max.
    other_particle: u32,
}

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

fn main() {
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
        .add_plugins(PlayerPlugin)
        .init_resource::<Controls>()
        .init_resource::<Constants>()
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
            (init_extract_kernel, init_copy_kernel, init_step_kernel),
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
    let scene =
        ron::de::from_reader::<_, fracture::Scene>(File::open("scene.ron").unwrap()).unwrap();

    let mesh = meshes.add(Sphere::new(0.5));

    for object in &scene.objects {
        let material = materials.add(StandardMaterial {
            base_color: object.color,
            ..default()
        });
        let fixed_material = materials.add(StandardMaterial {
            base_color: object.color.mix(&Color::BLACK, 0.5),
            ..default()
        });

        for &index in &object.particles {
            let particle = &scene.particles[index as usize];
            commands
                .spawn(PbrBundle {
                    mesh: mesh.clone(),
                    material: if particle.fixed {
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

    println!("Num particles: {}", scene.particles.len());

    let particles = scene
        .particles
        .iter()
        .map(|p| Particle {
            position: lv(p.position),
            next_position: lv(p.position),
            velocity: lv(p.velocity),
            next_velocity: lv(p.velocity),
            rest_position: lv(p.position),
            bond_start: p.bond_start,
            bond_count: p.bond_count,
            fixed: p.fixed,
        })
        .collect::<Vec<_>>();
    let bonds = scene
        .bonds
        .iter()
        .map(|bond| Bond {
            other_particle: bond.other_particle,
        })
        .collect::<Vec<_>>();
    commands.insert_resource(Buffers {
        domain: StaticDomain::<1>::new(particles.len() as u32),
        particles: device.create_buffer_from_slice(&particles),
        bonds: device.create_buffer_from_slice(&bonds),
        rendered_positions: device.create_buffer(particles.len()),
        rendered_positions_host: Arc::new(Mutex::new(
            scene.particles.iter().map(|p| lv(p.position)).collect(),
        )),
    });
}

#[kernel]
fn step_kernel(
    device: Res<LuisaDevice>,
    buffers: Res<Buffers>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    let gravity = lv(constants.gravity);
    Kernel::build(&device, &buffers.domain, &|index| {
        let particle = buffers.particles.read(*index).var();
        if particle.fixed {
            return;
        }
        let force = gravity.var();
        *force -= particle.velocity * constants.air_friction;
        for bond in **particle.bond_start..particle.bond_start + particle.bond_count {
            let index = buffers.bonds.read(bond).other_particle;
            if index == u32::MAX {
                continue;
            }
            let other = buffers.particles.read(index);
            let delta = other.position - particle.position;
            let delta_v = other.velocity - particle.velocity;
            let length = delta.norm();
            let rest_length = (other.rest_position - particle.rest_position).norm();
            if length > constants.breaking_distance * rest_length {
                buffers.bonds.write(
                    bond,
                    Bond {
                        other_particle: u32::MAX,
                    },
                );
                continue;
            }
            let dir = delta / length;
            let length_ratio_sq = (length / rest_length).sqr();
            let force_mag = (length_ratio_sq - 1.0 / length_ratio_sq) * constants.spring_constant
                + delta_v.dot(dir) * constants.damping_constant;
            *force += dir * force_mag;
        }
        *particle.next_velocity = particle.velocity + force * constants.dt;
        *particle.next_position = particle.position + particle.next_velocity * constants.dt;
        buffers.particles.write(*index, **particle);
    })
}
#[kernel]
fn extract_kernel(device: Res<LuisaDevice>, buffers: Res<Buffers>) -> Kernel<fn()> {
    Kernel::build(&device, &buffers.domain, &|index| {
        let particle = buffers.particles.read(*index);
        buffers.rendered_positions.write(*index, particle.position);
    })
}
#[kernel]
fn copy_kernel(device: Res<LuisaDevice>, buffers: Res<Buffers>) -> Kernel<fn()> {
    Kernel::build(&device, &buffers.domain, &|index| {
        let particle = buffers.particles.read(*index).var();
        *particle.velocity = particle.next_velocity;
        *particle.position = particle.next_position;
        buffers.particles.write(*index, **particle);
    })
}

fn step(
    device: Res<LuisaDevice>,
    constants: Res<Constants>,
    controls: Res<Controls>,
    buffers: Res<Buffers>,
) {
    let step = controls.running.then(|| {
        (0..constants.substeps)
            .map(|_| (step_kernel.dispatch(), copy_kernel.dispatch()).chain())
            .collect::<Vec<_>>()
            .chain()
    });
    let commands = (
        step,
        extract_kernel.dispatch(),
        buffers
            .rendered_positions
            .copy_to_shared(&buffers.rendered_positions_host),
    )
        .chain();
    ComputeGraph::new(&device).add(commands).execute();
}

#[derive(Debug, Clone, Copy, Resource)]
struct Constants {
    substeps: u32,
    dt: f32,
    gravity: Vec3,
    air_friction: f32,
    breaking_distance: f32,
    spring_constant: f32,
    damping_constant: f32,
}
impl Default for Constants {
    fn default() -> Self {
        Self {
            substeps: 10,
            dt: 1000.0 / 600.0,
            gravity: Vec3::new(0.0, -0.000002, 0.0),
            air_friction: 0.0,
            breaking_distance: 1.005,
            spring_constant: 0.001,
            damping_constant: 0.001,
        }
    }
}

#[derive(Debug, Resource)]
struct Buffers {
    domain: StaticDomain<1>,
    particles: Buffer<Particle>,
    bonds: Buffer<Bond>,
    rendered_positions: Buffer<LVec3<f32>>,
    rendered_positions_host: Arc<Mutex<Vec<LVec3<f32>>>>,
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
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            slice: false,
            running: false,
            slice_position: 0.01, // mostly to get r-a to shut up.
        }
    }
}

fn update_ui(mut contexts: EguiContexts, mut controls: ResMut<Controls>) {
    egui::Window::new("Controls").show(contexts.ctx_mut(), |ui| {
        ui.checkbox(&mut controls.running, "Running");
        ui.checkbox(&mut controls.slice, "Slice");
        ui.add(
            egui::Slider::new(&mut controls.slice_position, -10.0..=10.0).text("Slice Position"),
        );
    });
}

fn update_render(
    controls: Res<Controls>,
    buffers: Res<Buffers>,
    mut query: Query<(&ObjectParticle, &mut Transform, &mut Visibility)>,
) {
    let rendered_positions = buffers.rendered_positions_host.lock();

    for (particle, mut transform, mut visible) in query.iter_mut() {
        if controls.slice && transform.translation.x > controls.slice_position {
            *visible = Visibility::Hidden;
        } else {
            *visible = Visibility::Visible;
        }
        // Update nalgebra eventually.
        let pos = rendered_positions[particle.index as usize];
        transform.translation = Vec3::new(pos.x, pos.y, pos.z);
    }
}
