use std::f32::consts::PI;

use cosserat::{compute_coefficients, CosseratPbdInputs};
use utils::{step_pos_ang, Mat4x3, Vec3, Vec4};

use super::*;

#[derive(Debug, Resource)]
pub struct Particles {
    pub domain: StaticDomain<1>,
    pub linpos: Buffer<Vec3>,
    pub angpos: Buffer<Vec4>,
    pub linvel: Buffer<Vec3>,
    pub angvel: Buffer<Vec3>,

    pub last_linpos: Buffer<Vec3>,
    pub last_angpos: Buffer<Vec4>,

    pub bond_start: Buffer<u32>,
    pub bond_count: Buffer<u32>,
    // Used by BreakingModel::TotalStress only.
    pub broken: Option<Buffer<bool>>,
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
    let scale = constants.grid_scale();
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
    let (bend_twist_coeff, stretch_shear_coeff) = compute_coefficients(
        constants.dt,
        constants.bond_radius,
        constants.young_modulus,
        constants.shear_modulus,
    );
    let (force_bend_twist_coeff, force_stretch_shear_coeff) = if let BreakingModel::Stress {
        young_modulus,
        shear_modulus,
        ..
    } = constants.breaking_model
    {
        let young_modulus = if young_modulus == 0.0 {
            constants.young_modulus
        } else {
            young_modulus
        };
        let shear_modulus = if shear_modulus == 0.0 {
            constants.shear_modulus
        } else {
            shear_modulus
        };
        compute_coefficients(
            constants.dt,
            constants.bond_radius,
            young_modulus,
            shear_modulus,
        )
    } else {
        (Vec3::splat(0.0), Vec3::splat(0.0))
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

        let linvel_delta = Vec3::splat(0.0).var();
        let angvel_delta = Vec3::splat(0.0).var();

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

        let stress = 0.0_f32.var();

        for i in 0.expr()..bond_count {
            let bond = bond_start + i;
            let other = bonds.other_particle.read(bond);
            if other == u32::MAX {
                continue;
            }

            if let BreakingModel::TotalStress { .. } = constants.breaking_model {
                if particles.bond_count.read(other) == 0 {
                    bonds.other_particle.write(bond, u32::MAX);
                    continue;
                }
            }

            let other_linpos = particles.linpos.read(other);
            let other_angpos = particles.angpos.read(other);
            let pj = other_linpos;
            let qj = other_angpos;

            let pdiff = pj - pi;
            let qdiff = qj - qi;

            let length = bonds.length.read(bond);

            let current_length = pdiff.length();

            if let BreakingModel::Distance {
                max: breaking_distance,
                min: min_breaking_distance,
                angle: breaking_angle,
            } = constants.breaking_model
            {
                let l = current_length / length;
                if (breaking_distance != 0.0 && l > breaking_distance)
                    || (min_breaking_distance != 0.0 && l < min_breaking_distance)
                    || (breaking_angle != 0.0 && qdiff.norm() > breaking_angle)
                {
                    bonds.other_particle.write(bond, u32::MAX);
                    continue;
                }
            }

            *active_bonds += 1;

            let mj = particles.mass.read(other);
            let moj = 2.0_f32 / 5.0 * mj * constants.particle_radius.powi(2);

            let qrest = bonds.rest_rotation.read(bond);

            // TODO: Add forces so can do breaking model better.
            let outputs = cosserat::compute_pbd(CosseratPbdInputs {
                length,
                qrest,
                g,
                m: [mi, mj],
                mo: [moi, moj],
                pdiff,
                q: [qi, qj],
                qdiff,
                bend_twist_coeff,
                stretch_shear_coeff,
                force_bend_twist_coeff,
                force_stretch_shear_coeff,
            });

            if let BreakingModel::Stress {
                normal: max_normal_stress,
                shear: max_shear_stress,
                ..
            } = constants.breaking_model
            {
                let normal = pdiff / current_length;
                let normal_force = outputs.se_lin_force.dot(normal);
                let shear_force = (outputs.se_lin_force - normal_force * normal).norm();
                let twist_torque = outputs.bt_ang_force.dot(normal);
                let bend_torque = (outputs.bt_ang_force - twist_torque * normal).norm();

                // Can invert to make some weird elastic effects.
                let normal_stress = normal_force
                    / (5.0_f32 / 6.0 * PI * constants.bond_radius.powi(2))
                    + bend_torque * constants.bond_radius
                        / (PI * constants.bond_radius.powi(4) / 4.0);
                let shear_stress = 4.0 * shear_force / (3.0 * PI * constants.bond_radius.powi(2))
                    + twist_torque.abs() * constants.bond_radius
                        / (PI * constants.bond_radius.powi(4) / 2.0);

                if normal_stress > max_normal_stress || shear_stress > max_shear_stress {
                    bonds.other_particle.write(bond, u32::MAX);
                    // NOTE: We have an assymmetry in the bond breaking logic, but that should be corrected.
                    continue;
                }
            }

            *linvel_delta += outputs.se_lin_delta;
            *angvel_delta += outputs.se_ang_delta + outputs.bt_ang_delta;

            if let BreakingModel::TotalStress { .. } = constants.breaking_model {
                *stress += outputs.se_lin_delta.dot(pdiff);
            }
        }

        let collision_linvel_delta = Vec3::splat(0.0).var();

        let active_collisions = 0_u32.var();

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

                    *active_collisions += 1;
                    let delta = mi.recip() * (2.0 * constants.collision_particle_radius - pnorm)
                        / (mi.recip() + mj.recip() + collision_stiffness.recip());
                    *collision_linvel_delta -= delta * n;

                    if let BreakingModel::TotalStress {
                        ignore_collision: false,
                        ..
                    } = constants.breaking_model
                    {
                        *stress += delta;
                    }
                }
            }
        });

        let cosserat_step = match constants.cosserat_step {
            ConstraintStepModel::Factor(f) => f.expr(),
            ConstraintStepModel::StartingBondCount => luisa::max(bond_count, 1).cast_f32().recip(),
            ConstraintStepModel::CurrentBondCount => luisa::max(active_bonds, 1).cast_f32().recip(),
            ConstraintStepModel::CollisionCount => {
                luisa::max(active_collisions, 1).cast_f32().recip()
            }
        };
        let collision_step = match constants.collision_step {
            ConstraintStepModel::Factor(f) => f.expr(),
            ConstraintStepModel::StartingBondCount => luisa::max(bond_count, 1).cast_f32().recip(),
            ConstraintStepModel::CurrentBondCount => luisa::max(active_bonds, 1).cast_f32().recip(),
            ConstraintStepModel::CollisionCount => {
                luisa::max(active_collisions, 1).cast_f32().recip()
            }
        };

        particles.linvel.write(
            *index,
            particles.linvel.read(*index)
                + linvel_delta * cosserat_step
                + collision_linvel_delta * collision_step,
        );

        particles.angvel.write(
            *index,
            particles.angvel.read(*index) + angvel_delta * cosserat_step,
        );

        if let BreakingModel::TotalStress {
            max: max_stress, ..
        } = constants.breaking_model
        {
            if stress > max_stress {
                particles.broken.as_ref().unwrap().write(*index, true);
            }
        }
    })
}

#[kernel(init(pub))]
pub fn predict_kernel(particles: Res<Particles>, constants: Res<Constants>) -> Kernel<fn()> {
    Kernel::build(&particles.domain, &|index| {
        let linvel = if constants.gravity == bevy::math::Vec3::ZERO
            || particles.mass.read(*index) == f32::INFINITY
        {
            particles.linvel.read(*index)
        } else {
            let lv = particles.linvel.read(*index)
                + Vec3::from(constants.gravity * constants.dt * constants.dt);
            particles.linvel.write(*index, lv);
            lv
        };
        let linpos = particles.linpos.read(*index);
        let next_linpos = linpos + linvel;
        particles.last_linpos.write(*index, linpos);
        particles.linpos.write(*index, next_linpos);

        let angvel = particles.angvel.read(*index);
        let angpos = particles.angpos.read(*index);
        let next_angpos = step_pos_ang(angpos, angvel);
        particles.last_angpos.write(*index, angpos);
        particles.angpos.write(*index, next_angpos);
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

        if let Some(broken) = &particles.broken {
            if broken.read(*index) {
                particles.bond_count.write(*index, 0);
            }
        }
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
    let size_i = LVec3::from(size.as_ivec3());

    let position = position.rem_euclid(size_i).cast_u32();
    position.x + size.x * (position.y + size.y * position.z)
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
            constants.grid_scale(),
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
        let cell = grid_cell(position, constants.grid_size, constants.grid_scale());
        let offset = grid.offset.read(cell) + grid.count.atomic_ref(cell).fetch_add(1);
        grid.particles.write(offset, *index);
    })
}
