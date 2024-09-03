use std::f32::consts::PI;

use utils::{step_pos_ang, Mat4x3};

use super::*;

type Vec3 = LVec3<f32>;
type Vec4 = LVec4<f32>;

#[derive(Debug, Resource)]
pub struct Particles {
    pub domain: StaticDomain<1>,
    pub linpos: Buffer<Vec3>,
    pub angpos: Buffer<Vec4>,
    pub linvel: Buffer<Vec3>,
    pub angvel: Buffer<Vec3>,

    pub last_linpos: Buffer<Vec3>,
    pub last_angpos: Buffer<Vec4>,
    pub last_linvel: Buffer<Vec3>,
    pub last_angvel: Buffer<Vec3>,

    pub bond_start: Buffer<u32>,
    pub bond_count: Buffer<u32>,
    pub mass: Buffer<f32>,
}

#[derive(Debug, Resource)]
pub struct Bonds {
    pub other_particle: Buffer<u32>,
    pub length: Buffer<f32>,
    pub rest_rotation: Buffer<Vec4>,
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
pub fn neighbors(grid: &Grid, constants: &Constants, position: Expr<Vec3>, f: impl Fn(Expr<u32>)) {
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
    particles: Res<Particles>,
    bonds: Res<Bonds>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    let dt2 = constants.dt * constants.dt;
    let bend_twist_coeff = {
        let i = PI * constants.bond_radius.powi(4) / 4.0;
        let j = PI * constants.bond_radius.powi(4) / 2.0;
        Vec3::new(
            constants.young_modulus * i * dt2,
            constants.young_modulus * i * dt2,
            constants.shear_modulus * j * dt2,
        )
    };

    let stretch_shear_coeff = {
        let s = PI * constants.bond_radius.powi(2);
        let a = 5.0_f32 / 6.0 * s;
        Vec3::new(
            constants.shear_modulus * a * dt2,
            constants.shear_modulus * a * dt2,
            constants.young_modulus * s * dt2,
        )
    };

    Kernel::build(&particles.domain, &|index| {
        let mi = particles.mass.read(*index);
        if mi == f32::INFINITY {
            return;
        }
        let moi = 2.0_f32 / 5.0 * mi * constants.particle_radius.powi(2);

        let bond_start = particles.bond_start.read(*index);
        let bond_count = particles.bond_count.read(*index);

        let linpos = particles.linpos.read(*index);
        let angpos = particles.angpos.read(*index);

        let linvel_delta = Vec3::splat_expr(0.0).var();
        let angvel_delta = Vec3::splat_expr(0.0).var();

        let pi = linpos;
        let qi = angpos;

        let g = {
            let q = qi / 2.0;
            Mat4x3::expr(
                Vec4::expr(q.w, -q.z, q.y, -q.x),
                Vec4::expr(q.z, q.w, -q.x, -q.y),
                Vec4::expr(-q.y, q.x, q.w, -q.z),
            )
        };

        let active_bonds = 0_u32.var();

        for bond in bond_start..bond_start + bond_count {
            let other = bonds.other_particle.read(bond);
            if other == u32::MAX {
                continue;
            }

            let other_linpos = particles.linpos.read(other);
            let pj = other_linpos;
            let pdiff = pj - pi;
            let length = bonds.length.read(bond);

            let current_length = pdiff.length();

            if current_length / length > constants.breaking_distance
                || (constants.min_breaking_distance != 0.0
                    && current_length / length < constants.min_breaking_distance)
            {
                bonds.other_particle.write(bond, u32::MAX);
                continue;
            }

            *active_bonds += 1;

            let mj = particles.mass.read(other);
            let moj = 2.0_f32 / 5.0 * mj * constants.particle_radius.powi(2);

            let other_angpos = particles.angpos.read(other);

            let qj = other_angpos;

            let qdiff = qj - qi;

            let length = bonds.length.read(bond);
            let qrest = bonds.rest_rotation.read(bond);

            let outputs = cosserat::compute_pbd(
                bend_twist_coeff,
                stretch_shear_coeff,
                length,
                qrest,
                g,
                [mi, mj],
                [moi, moj],
                pdiff,
                [qi, qj],
                qdiff,
            );

            *linvel_delta += outputs.se_lin_delta;
            *angvel_delta += outputs.se_ang_delta + outputs.bt_ang_delta;
        }

        // TODO: I can remove self-collisions by subtracting this collision when iterating over bonds.
        neighbors(&grid, &constants, linpos, |other| {
            if other != *index {
                let collision_stiffness = constants.collision_stiffness * dt2;
                let pj = particles.linpos.read(other);
                let mj = particles.mass.read(other);
                let pdiff = pj - pi;
                let pnorm = pdiff.length();
                if pnorm <= 2.0 * constants.collision_particle_radius {
                    let n = pdiff / pnorm;

                    *linvel_delta -=
                        mi.recip() * n * (2.0 * constants.collision_particle_radius - pnorm)
                            / (mi.recip() + mj.recip() + collision_stiffness.recip());
                }
            }
        });

        let constraint_step = match constants.constraint_step {
            ConstraintStepModel::Factor(f) => f.expr(),
            ConstraintStepModel::StartingBondCount => luisa::max(bond_count, 1).cast_f32().recip(),
            ConstraintStepModel::CurrentBondCount => luisa::max(active_bonds, 1).cast_f32().recip(),
        };

        particles.linvel.write(
            *index,
            particles.linvel.read(*index) + linvel_delta * constraint_step,
        );

        particles.angvel.write(
            *index,
            particles.angvel.read(*index) + angvel_delta * constraint_step,
        );
    })
}

#[kernel(init(pub))]
pub fn predict_kernel(particles: Res<Particles>) -> Kernel<fn()> {
    Kernel::build(&particles.domain, &|index| {
        let linvel = particles.linvel.read(*index);
        let linpos = particles.linpos.read(*index);
        let next_linpos = linpos + linvel;
        particles.last_linpos.write(*index, linpos);
        particles.linpos.write(*index, next_linpos);
        particles.last_linvel.write(*index, linvel);

        let angvel = particles.angvel.read(*index);
        let angpos = particles.angpos.read(*index);
        let next_angpos = step_pos_ang(angpos, angvel);
        particles.last_angpos.write(*index, angpos);
        particles.angpos.write(*index, next_angpos);
        particles.last_angvel.write(*index, angvel);
    })
}

#[kernel(init(pub))]
pub fn solve_update_kernel(particles: Res<Particles>) -> Kernel<fn()> {
    Kernel::build(&particles.domain, &|index| {
        let linvel = particles.linvel.read(*index);
        let linpos = particles.last_linpos.read(*index) + linvel;
        particles.linpos.write(*index, linpos);

        let angvel = particles.angvel.read(*index);
        let angpos = step_pos_ang(particles.last_angpos.read(*index), angvel);
        particles.angpos.write(*index, angpos);
    })
}

pub fn step(constants: Res<Constants>, controls: Res<Controls>, ev: Res<ButtonInput<KeyCode>>) {
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
                solve_update_kernel.dispatch(),
            )
                .chain()
        })
        .collect::<Vec<_>>()
        .chain();
    #[cfg(feature = "timed")]
    {
        let timings = ComputeGraph::new().add(commands).execute_timed();
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
        ComputeGraph::new().add(commands).execute();
    }
}

#[kernel(init(pub))]
pub fn reset_grid_kernel(grid: Res<Grid>) -> Kernel<fn()> {
    Kernel::build(&grid.domain, &|index| {
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
pub fn grid_cell(position: Expr<Vec3>, size: UVec3, scale: f32) -> Expr<u32> {
    let position = position / scale;
    grid_cell_index(position.floor().cast_i32(), size)
}

#[kernel(init(pub))]
pub fn count_kernel(
    particles: Res<Particles>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    Kernel::build(&particles.domain, &|index| {
        let cell = grid_cell(
            particles.linpos.read(*index),
            constants.grid_size,
            constants.grid_scale,
        );
        grid.count.atomic_ref(cell).fetch_add(1);
    })
}

#[kernel(init(pub))]
pub fn compute_offset_kernel(grid: Res<Grid>) -> Kernel<fn()> {
    Kernel::build(&grid.domain, &|index| {
        let count = grid.count.read(*index);
        grid.offset
            .write(*index, grid.next_block.atomic_ref(0).fetch_add(count));
        grid.count.write(*index, 0);
    })
}

#[kernel(init(pub))]
pub fn add_particle_kernel(
    particles: Res<Particles>,
    grid: Res<Grid>,
    constants: Res<Constants>,
) -> Kernel<fn()> {
    Kernel::build(&particles.domain, &|index| {
        let position = particles.linpos.read(*index);
        let cell = grid_cell(position, constants.grid_size, constants.grid_scale);
        let offset = grid.offset.read(cell) + grid.count.atomic_ref(cell).fetch_add(1);
        grid.particles.write(offset, *index);
    })
}
