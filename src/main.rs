use std::fs::File;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::settings::{PowerPreference, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy_egui::{EguiContexts, EguiPlugin};
use bevy_flycam::{FlyCam, MovementSettings, NoCameraPlayerPlugin};
use bevy_sefirot::kernel;
use bevy_sefirot::luisa::{InitKernel, LuisaDevice, LuisaPlugin};
use luisa::lang::types::vector::Vec3 as LVec3;
use luisa_compute::DeviceType;
use parking_lot::Mutex;
use sefirot::graph::{AsNodes as AsNodesExt, ComputeGraph};
use sefirot::mapping::buffer::StaticDomain;
use sefirot::prelude::*;

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
        .add_plugins(NoCameraPlayerPlugin)
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
            (
                init_count_kernel,
                init_reset_grid_kernel,
                init_add_particle_kernel,
                init_compute_offset_kernel,
                init_copy_kernel,
                init_step_kernel,
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
    constants: Res<Constants>,
) {
    let scene =
        ron::de::from_reader::<_, fracture::Scene>(File::open("collide.ron").unwrap()).unwrap();

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

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(30.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        FlyCam,
    ));

    println!("Num particles: {}", scene.particles.len());

    let particles = scene.particles;
    let l = particles.len();

    let particles = Particles {
        domain: StaticDomain::<1>::new(l as u32),
        position: device.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        next_position: device.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        velocity: device.create_buffer_from_fn(l, |i| lv(particles[i].velocity)),
        next_velocity: device.create_buffer_from_fn(l, |i| lv(particles[i].velocity)),
        rest_position: device.create_buffer_from_fn(l, |i| lv(particles[i].position)),
        bond_start: device.create_buffer_from_fn(l, |i| particles[i].bond_start),
        bond_count: device.create_buffer_from_fn(l, |i| particles[i].bond_count),
        fixed: device.create_buffer_from_fn(l, |i| particles[i].fixed),
        rendered_positions_host: Arc::new(Mutex::new(
            particles.iter().map(|p| lv(p.position)).collect(),
        )),
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
    commands.insert_resource(particles);
    commands.insert_resource(bonds);
    commands.insert_resource(grid);
}

#[tracked]
fn neighbors(
    grid: &Grid,
    constants: &Constants,
    position: Expr<LVec3<f32>>,
    f: impl Fn(Expr<u32>),
) {
    let size = constants.grid_size;
    let scale = constants.grid_scale;
    let position = (position / scale).floor().cast_i32();
    for i in -1..=1 {
        for j in -1..=1 {
            for k in -1..=1 {
                let offset = LVec3::expr(i, j, k);
                let cell = grid_cell_index(position + offset, size);
                let offset = grid.offset.read(cell);
                let count = grid.count.read(cell);
                for i in 0.expr()..count {
                    f(grid.particles.read(offset + i));
                }
            }
        }
    }
}

#[kernel]
fn step_kernel(
    device: Res<LuisaDevice>,
    particles: Res<Particles>,
    bonds: Res<Bonds>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    let gravity = lv(constants.gravity);
    Kernel::build(&device, &particles.domain, &|index| {
        if particles.fixed.read(*index) {
            return;
        }
        let velocity = particles.velocity.read(*index);
        let position = particles.position.read(*index);
        let rest_position = particles.rest_position.read(*index);
        let bond_start = particles.bond_start.read(*index);
        let bond_count = particles.bond_count.read(*index);
        let force = gravity.var();
        *force -= velocity * constants.air_friction;
        for bond in bond_start..bond_start + bond_count {
            let other = bonds.other_particle.read(bond);
            if other == u32::MAX {
                continue;
            }
            let other_position = particles.position.read(other);
            let other_velocity = particles.velocity.read(other);
            let other_rest_position = particles.rest_position.read(other);
            let delta = other_position - position;
            let delta_v = other_velocity - velocity;
            let length = delta.norm();
            let rest_length = (other_rest_position - rest_position).norm();
            if length > constants.breaking_distance * rest_length {
                bonds.other_particle.write(bond, u32::MAX);
                continue;
            }
            let dir = delta / length;
            let force_mag = (length - rest_length) * constants.spring_constant
                + delta_v.dot(dir) * constants.damping_constant;
            *force += dir * force_mag;
        }
        neighbors(&grid, &constants, position, |other| {
            if other != *index {
                let other_position = particles.position.read(other);
                let delta = other_position - position;
                let length = delta.norm();
                let penetration = length - constants.particle_radius * 2.0;
                if penetration < 0.0 {
                    let dir = delta / length;
                    let force_mag = penetration * constants.collision_constant;
                    *force += dir * force_mag;
                }
            }
        });
        let next_velocity = velocity + force * constants.dt;
        let next_position = position + next_velocity * constants.dt;
        particles.next_velocity.write(*index, next_velocity);
        particles.next_position.write(*index, next_position);
    })
}
#[kernel]
fn copy_kernel(device: Res<LuisaDevice>, particles: Res<Particles>) -> Kernel<fn()> {
    Kernel::build(&device, &particles.domain, &|index| {
        particles
            .position
            .write(*index, particles.next_position.read(*index));
        particles
            .velocity
            .write(*index, particles.next_velocity.read(*index));
    })
}

fn step(
    device: Res<LuisaDevice>,
    constants: Res<Constants>,
    controls: Res<Controls>,
    particles: Res<Particles>,
) {
    let step = controls.running.then(|| {
        (0..constants.substeps)
            .map(|_| {
                (
                    count_kernel.dispatch(),
                    reset_grid_kernel.dispatch(),
                    compute_offset_kernel.dispatch(),
                    add_particle_kernel.dispatch(),
                    step_kernel.dispatch(),
                    copy_kernel.dispatch(),
                )
                    .chain()
            })
            .collect::<Vec<_>>()
            .chain()
    });
    let commands = (
        step,
        particles
            .position
            .copy_to_shared(&particles.rendered_positions_host),
    )
        .chain();
    #[cfg(feature = "timed")]
    {
        let timings = ComputeGraph::new(&device).add(commands).execute_timed();
        let step_times = timings
            .iter()
            .filter_map(|(name, time)| (name == "step_kernel").then_some(time))
            .collect::<Vec<_>>();
        if !step_times.is_empty() {
            println!(
                "Step time: {:?}",
                step_times.iter().copied().copied().sum::<f32>() / step_times.len() as f32
            );
        }
    }
    #[cfg(not(feature = "timed"))]
    {
        ComputeGraph::new(&device).add(commands).execute();
    }
}

#[kernel]
fn reset_grid_kernel(device: Res<LuisaDevice>, grid: Res<Grid>) -> Kernel<fn()> {
    Kernel::build(&device, &grid.domain, &|index| {
        grid.count.write(*index, 0);
        if *index == 0 {
            grid.next_block.write(0, 0);
        }
    })
}

#[tracked]
fn grid_cell_index(position: Expr<LVec3<i32>>, size: UVec3) -> Expr<u32> {
    let size_i = LVec3::new(size.x as i32, size.y as i32, size.z as i32);

    let position = position.rem_euclid(size_i).cast_u32();
    position.y + size.x * (position.x + size.y * position.z)
}

#[tracked]
fn grid_cell(position: Expr<LVec3<f32>>, size: UVec3, scale: f32) -> Expr<u32> {
    let position = position / scale;
    grid_cell_index(position.floor().cast_i32(), size)
}

#[kernel]
fn count_kernel(
    device: Res<LuisaDevice>,
    particles: Res<Particles>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    Kernel::build(&device, &particles.domain, &|index| {
        let cell = grid_cell(
            particles.position.read(*index),
            constants.grid_size,
            constants.grid_scale,
        );
        grid.count.atomic_ref(cell).fetch_add(1);
    })
}

#[kernel]
fn compute_offset_kernel(device: Res<LuisaDevice>, grid: Res<Grid>) -> Kernel<fn()> {
    Kernel::build(&device, &grid.domain, &|index| {
        let count = grid.count.read(*index);
        grid.offset
            .write(*index, grid.next_block.atomic_ref(0).fetch_add(count));
    })
}

#[kernel]
fn add_particle_kernel(
    device: Res<LuisaDevice>,
    particles: Res<Particles>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    Kernel::build(&device, &particles.domain, &|index| {
        let position = particles.position.read(*index);
        let cell = grid_cell(position, constants.grid_size, constants.grid_scale);
        let offset = grid.offset.read(cell) + grid.count.atomic_ref(cell).fetch_add(1);
        grid.particles.write(offset, *index);
    })
}

#[derive(Debug, Clone, Copy, Resource)]
struct Constants {
    substeps: u32,
    dt: f32,
    gravity: Vec3,
    air_friction: f32,
    breaking_distance: f32,
    spring_constant: f32,
    collision_constant: f32,
    damping_constant: f32,
    grid_size: UVec3,
    grid_scale: f32,
    particle_radius: f32,
}
impl Default for Constants {
    fn default() -> Self {
        Self {
            substeps: 10,
            dt: 1.0 / 600.0,
            gravity: Vec3::ZERO, // Vec3::new(0.0, -0.000002, 0.0),
            air_friction: 0.0,
            breaking_distance: 1.02,
            spring_constant: 000.0,
            collision_constant: 1000.0,
            damping_constant: 00.0,
            grid_size: UVec3::splat(40),
            grid_scale: 1.0, // The particle diameter.
            particle_radius: 0.5,
        }
    }
}

#[derive(Debug, Resource)]
struct Particles {
    domain: StaticDomain<1>,
    position: Buffer<LVec3<f32>>,
    next_position: Buffer<LVec3<f32>>,
    velocity: Buffer<LVec3<f32>>,
    next_velocity: Buffer<LVec3<f32>>,
    rest_position: Buffer<LVec3<f32>>,
    bond_start: Buffer<u32>,
    bond_count: Buffer<u32>,
    fixed: Buffer<bool>,
    rendered_positions_host: Arc<Mutex<Vec<LVec3<f32>>>>,
}

#[derive(Debug, Resource)]
struct Bonds {
    other_particle: Buffer<u32>,
}

#[derive(Debug, Resource)]
struct Grid {
    domain: StaticDomain<1>,
    count: Buffer<u32>,
    offset: Buffer<u32>,
    particles: Buffer<u32>,
    // For atomics.
    next_block: Buffer<u32>,
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
    particles: Res<Particles>,
    mut query: Query<(&ObjectParticle, &mut Transform, &mut Visibility)>,
) {
    let rendered_positions = particles.rendered_positions_host.lock();

    for (particle, mut transform, mut visible) in query.iter_mut() {
        if controls.slice && transform.translation.x > controls.slice_position {
            *visible = Visibility::Hidden;
        } else {
            *visible = Visibility::Visible;
        }
        // Update nalgebra eventually.
        let pos = rendered_positions[particle.index as usize];
        if pos.x.is_infinite() || pos.y.is_infinite() || pos.z.is_infinite() {
            panic!("Infinite position");
        }
        transform.translation = Vec3::new(pos.x, pos.y, pos.z);
    }
}
