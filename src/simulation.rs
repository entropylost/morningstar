use super::*;

#[derive(Debug, Resource)]
pub struct Particles {
    pub domain: StaticDomain<1>,
    pub position: Buffer<LVec3<f32>>,
    pub next_position: Buffer<LVec3<f32>>,
    pub displacement: Buffer<LVec3<f32>>,
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
pub fn solve_kernel(
    device: Res<LuisaDevice>,
    particles: Res<Particles>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    Kernel::build(&device, &particles.domain, &|index| {
        let position = particles.next_position.read(*index);
        if particles.fixed.read(*index) {
            particles.displacement.write(*index, LVec3::splat_expr(0.0));
            return;
        }

        let displacement = LVec3::splat_expr(0.0_f32).var();

        neighbors(&grid, &constants, position, |other| {
            if other != *index {
                let length = delta.norm();
                let penetration = 2.0 * constants.particle_radius - length;
                if penetration > 0.0 {
                    let normal = delta / length;
                    *displacement += normal * penetration / 2.0;
                }
            }
        });

        particles.displacement.write(*index, displacement);
    })
}
#[kernel(init(pub))]
pub fn predict_kernel(device: Res<LuisaDevice>, particles: Res<Particles>) -> Kernel<fn()> {
    Kernel::build(&device, &particles.domain, &|index| {
        let last_pos = particles.position.read(*index);
        let pos = particles.next_position.read(*index) + particles.displacement.read(*index);
        let vel = pos - last_pos;
        let next_pos = pos + vel;
        particles.position.write(*index, pos);
        particles.next_position.write(*index, next_pos);
    })
}

pub fn step(
    device: Res<LuisaDevice>,
    constants: Res<Constants>,
    controls: Res<Controls>,
    // grid: Res<Grid>,
    // particles: Res<Particles>,
    ev: Res<ButtonInput<KeyCode>>,
) {
    if !controls.running && !ev.just_pressed(KeyCode::Period) {
        return;
    }
    let commands = (0..constants.substeps)
        .map(|_| {
            (
                predict_kernel.dispatch(),
                reset_grid_kernel.dispatch(),
                count_kernel.dispatch(),
                compute_offset_kernel.dispatch(),
                add_particle_kernel.dispatch(),
                solve_kernel.dispatch(),
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
    // println!("Positions: {:?}", particles.position.copy_to_vec());
    // println!(
    //     "Next Positions: {:?}",
    //     particles.next_position.copy_to_vec()
    // );
    // println!("Displacements: {:?}", particles.displacement.copy_to_vec());
    // println!("Grid Count: {:?}", grid.count.copy_to_vec());
    // println!("Offsets: {:?}", grid.offset.copy_to_vec());
    // println!("Particles: {:?}", grid.particles.copy_to_vec());
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
            particles.next_position.read(*index),
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
        let position = particles.next_position.read(*index);
        let cell = grid_cell(position, constants.grid_size, constants.grid_scale);
        let offset = grid.offset.read(cell) + grid.count.atomic_ref(cell).fetch_add(1);
        grid.particles.write(offset, *index);
    })
}
