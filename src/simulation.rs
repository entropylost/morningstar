use super::*;

#[derive(Debug, Resource)]
pub struct Particles {
    pub domain: StaticDomain<1>,
    pub position: Buffer<LVec3<f32>>,
    pub next_position: Buffer<LVec3<f32>>,
    pub velocity: Buffer<LVec3<f32>>,
    pub next_velocity: Buffer<LVec3<f32>>,
    pub rest_position: Buffer<LVec3<f32>>,
    pub bond_start: Buffer<u32>,
    pub bond_count: Buffer<u32>,
    pub fixed: Buffer<bool>,
}

#[derive(Debug, Resource)]
pub struct Bonds {
    pub other_particle: Buffer<u32>,
}

#[derive(Debug, Resource)]
pub struct Grid {
    pub domain: StaticDomain<1>,
    pub count: Buffer<u32>,
    pub offset: Buffer<u32>,
    pub particles: Buffer<u32>,
    // For atomics.
    pub next_block: Buffer<u32>,
}

#[tracked]
pub fn neighbors(
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

#[kernel(init(pub))]
pub fn step_kernel(
    device: Res<LuisaDevice>,
    particles: Res<Particles>,
    bonds: Res<Bonds>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    let gravity = lv(constants.gravity);
    Kernel::build(&device, &particles.domain, &|index| {
        if particles.fixed.read(*index) {
            let next_position =
                particles.position.read(*index) + particles.velocity.read(*index) * constants.dt;
            particles.next_position.write(*index, next_position);
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
            if length > constants.breaking_distance * rest_length
                || if constants.min_breaking_distance != 0.0 {
                    length < constants.min_breaking_distance * rest_length
                } else {
                    false.expr()
                }
            {
                bonds.other_particle.write(bond, u32::MAX);
                continue;
            }
            let dir = delta / length;

            let l = length / rest_length;

            let force_mag = delta_v.dot(dir) * constants.damping_constant
                + match constants.spring_model {
                    SpringModel::Linear => (l - 1.0) * constants.spring_constant,
                    SpringModel::Quadratic => (l - 1.0).sqr() * constants.spring_constant,
                    SpringModel::InvQuadratic => {
                        (l.sqr() - 1.0 / l.sqr()) * constants.spring_constant
                    }
                    SpringModel::Impulse(ImpulseConstants { bias, slop: _ }) => {
                        let bias_vel = -bias * (length - rest_length);

                        let normal_mass = if particles.fixed.read(other) {
                            1.0_f32.expr()
                        } else {
                            0.5.expr()
                        };
                        let impulse = (delta_v.dot(dir)) * normal_mass;
                        impulse / constants.dt / bond_count.cast_f32()
                            + (l - 1.0) * constants.spring_constant
                    }
                };
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
                    let force_mag = match constants.collision_model {
                        SpringModel::Linear => penetration * constants.collision_constant,
                        SpringModel::Quadratic => penetration.sqr() * constants.collision_constant,
                        SpringModel::InvQuadratic => {
                            let l = length / (constants.particle_radius * 2.0);
                            (l.sqr() - 1.0 / l.sqr()) * constants.collision_constant
                        }
                        SpringModel::Impulse(ImpulseConstants { bias, slop }) => {
                            let other_velocity = particles.velocity.read(other);
                            let dv = other_velocity - velocity;
                            let bias_vel = bias * luisa::min(slop - penetration, 0.0);

                            let normal_mass = if particles.fixed.read(other) {
                                1.0_f32.expr()
                            } else {
                                0.5.expr()
                            };
                            let impulse = (dv.dot(dir) + bias_vel) * normal_mass;
                            luisa::min(impulse, 0.0) / constants.dt
                        }
                    };
                    *force += dir * force_mag;
                }
            }
        });
        let next_velocity = velocity + force * constants.dt;
        let next_velocity = next_velocity.var();
        let next_position = position + next_velocity * constants.dt;
        let next_position = next_position.var();
        if constants.floor != f32::NEG_INFINITY {
            if next_position.y < constants.floor {
                *next_position.y = constants.floor;
                *next_velocity.y = luisa::max(0.0, -constants.floor_restitution * next_velocity.y);
            }
        }
        particles.next_velocity.write(*index, next_velocity);
        particles.next_position.write(*index, next_position);
    })
}
#[kernel(init(pub))]
pub fn copy_kernel(device: Res<LuisaDevice>, particles: Res<Particles>) -> Kernel<fn()> {
    Kernel::build(&device, &particles.domain, &|index| {
        particles
            .position
            .write(*index, particles.next_position.read(*index));
        particles
            .velocity
            .write(*index, particles.next_velocity.read(*index));
    })
}

pub fn step(
    device: Res<LuisaDevice>,
    constants: Res<Constants>,
    controls: Res<Controls>,
    ev: Res<ButtonInput<KeyCode>>,
) {
    if !controls.running && !ev.just_pressed(KeyCode::Period) {
        return;
    }
    let commands = (0..constants.substeps)
        .map(|_| {
            (
                reset_grid_kernel.dispatch(),
                count_kernel.dispatch(),
                compute_offset_kernel.dispatch(),
                add_particle_kernel.dispatch(),
                step_kernel.dispatch(),
                copy_kernel.dispatch(),
            )
                .chain()
        })
        .collect::<Vec<_>>()
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

#[kernel(init(pub))]
pub fn reset_grid_kernel(device: Res<LuisaDevice>, grid: Res<Grid>) -> Kernel<fn()> {
    Kernel::build(&device, &grid.domain, &|index| {
        grid.count.write(*index, 0);
        if *index == 0 {
            grid.next_block.write(0, 0);
        }
    })
}

#[tracked]
pub fn grid_cell_index(position: Expr<LVec3<i32>>, size: UVec3) -> Expr<u32> {
    let size_i = LVec3::new(size.x as i32, size.y as i32, size.z as i32);

    let position = position.rem_euclid(size_i).cast_u32();
    position.y + size.x * (position.x + size.y * position.z)
}

#[tracked]
pub fn grid_cell(position: Expr<LVec3<f32>>, size: UVec3, scale: f32) -> Expr<u32> {
    let position = position / scale;
    grid_cell_index(position.floor().cast_i32(), size)
}

#[kernel(init(pub))]
pub fn count_kernel(
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

#[kernel(init(pub))]
pub fn compute_offset_kernel(device: Res<LuisaDevice>, grid: Res<Grid>) -> Kernel<fn()> {
    Kernel::build(&device, &grid.domain, &|index| {
        let count = grid.count.read(*index);
        grid.offset
            .write(*index, grid.next_block.atomic_ref(0).fetch_add(count));
        grid.count.write(*index, 0);
    })
}

#[kernel(init(pub))]
pub fn add_particle_kernel(
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
