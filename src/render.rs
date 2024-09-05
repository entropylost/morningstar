use bevy::render::render_asset::RenderAssetUsages;
use bevy::winit::WinitWindows;
use image::buffer::ConvertBuffer;
use image::{Rgba32FImage, RgbaImage};
use luisa::lang::types::vector::Mat3 as LMat3;
use utils::{conjugate, qapply, Vec3, Vec4};

use super::*;

#[derive(Debug, Component)]
pub struct RenderSprite;

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct Slice {
    pub normal: Vec3,
    pub position: Vec3,
}

#[derive(Debug, Clone, Resource)]
pub struct RenderConstants {
    pub grid_size: UVec3,
    pub grid_scale: f32,
    pub max_radius: f32,
    // For adding external walls.
    pub additional_slices: [Slice; 6],
}

#[derive(Debug)]
pub struct RenderGrid {
    pub domain: StaticDomain<1>,
    pub count: Buffer<u32>,
    pub occupance: Buffer<bool>,
    pub offset: Buffer<u32>,
    pub particles: Buffer<u32>,
    // For atomics.
    pub next_block: Buffer<u32>,
}

#[derive(Debug, Resource)]
pub struct RenderData {
    // Internal bonds don't need to be checked (?) so can optimize there a bit.
    // broken_bonds: Buffer<u32>,
    pub starting_positions: Buffer<Vec3>,

    pub screen: Buffer<Vec4>,
    pub screen_domain: StaticDomain<2>,

    pub grid: RenderGrid,

    pub bounds: (bevy::math::Vec3, bevy::math::Vec3),
}

#[kernel(init(pub))]
fn count_kernel(
    constants: Res<RenderConstants>,
    particles: Res<simulation::Particles>,
    data: Res<RenderData>,
) -> Kernel<fn()> {
    Kernel::build(&particles.domain, &|index| {
        if particles.mass.read(*index) == f32::INFINITY {
            return;
        }
        let pos = particles.linpos.read(*index);
        let min = (pos / constants.grid_scale).floor().cast_i32() - 1;
        let max = (pos / constants.grid_scale).floor().cast_i32() + 2;

        // let min = ((pos - constants.max_radius) / constants.grid_scale)
        //     .floor()
        //     .cast_i32()
        //     - 1;
        // let max = ((pos + constants.max_radius) / constants.grid_scale)
        //     .ceil()
        //     .cast_i32()
        //     + 1;
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let v = LVec3::<i32>::expr(x, y, z)
                        .rem_euclid(LVec3::from(constants.grid_size.as_ivec3()))
                        .cast_u32();
                    let cell = v.x + constants.grid_size.x * (v.y + constants.grid_size.y * v.z);
                    data.grid.count.atomic_ref(cell).fetch_add(1);
                    data.grid.occupance.write(cell, true);
                }
            }
        }
    })
}

#[kernel(init(pub))]
pub fn compute_offset_kernel(data: Res<RenderData>) -> Kernel<fn()> {
    let grid = &data.grid;
    Kernel::build(&grid.domain, &|index| {
        let count = grid.count.read(*index);
        grid.offset
            .write(*index, grid.next_block.atomic_ref(0).fetch_add(count));
        grid.count.write(*index, 0);
    })
}

#[kernel(init(pub))]
pub fn add_particle_kernel(
    constants: Res<RenderConstants>,
    particles: Res<simulation::Particles>,
    data: Res<RenderData>,
) -> Kernel<fn()> {
    let grid = &data.grid;
    Kernel::build(&particles.domain, &|index| {
        if particles.mass.read(*index) == f32::INFINITY {
            return;
        }
        let pos = particles.linpos.read(*index);
        let min = (pos / constants.grid_scale).floor().cast_i32() - 1;
        let max = (pos / constants.grid_scale).floor().cast_i32() + 2;

        // ((pos - constants.max_radius) / constants.grid_scale)
        //     .floor()
        //     .cast_i32()
        //     - 1;
        // let max = ((pos + constants.max_radius) / constants.grid_scale)
        //     .ceil()
        //     .cast_i32()
        //     + 1;

        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let v = LVec3::<i32>::expr(x, y, z)
                        .rem_euclid(LVec3::from(constants.grid_size.as_ivec3()))
                        .cast_u32();
                    let cell = v.x + constants.grid_size.x * (v.y + constants.grid_size.y * v.z);
                    let offset = grid.offset.read(cell) + grid.count.atomic_ref(cell).fetch_add(1);
                    grid.particles.write(offset, *index);
                }
            }
        }
    })
}

pub fn render(
    mut running: Local<bool>,
    data: Res<RenderData>,
    camera: Query<&Transform, With<Camera3d>>,
    input: Res<ButtonInput<KeyCode>>,
    mut images: ResMut<Assets<Image>>,
    mut sprite: Query<&mut Handle<Image>, With<RenderSprite>>,
) {
    if input.just_pressed(KeyCode::KeyR) {
        *running = !*running;
    }
    if !*running {
        return;
    }
    let transform = camera.single();
    let pos = transform.translation;
    // println!("{:?}", pos);
    let ratio = (1.15_f32 / 2.0).tan() / (data.screen_domain.height() as f32 / 2.0);
    let view = Mat3::from_cols(
        transform.right() * ratio,
        transform.down() * ratio,
        transform.forward().as_vec3(),
    );
    // println!("{:?}", view);
    (
        data.grid.count.fill(0),
        data.grid.next_block.fill(0),
        data.grid.occupance.fill(false),
        count_kernel.dispatch(),
        compute_offset_kernel.dispatch(),
        add_particle_kernel.dispatch(),
        trace_kernel.dispatch(
            &LMat3::from(view),
            &LVec3::from(pos),
            &LVec3::from(data.bounds.0),
            &LVec3::from(data.bounds.1),
        ),
    )
        .chain()
        .execute();
    let screen = data.screen.copy_to_vec();
    let image = screen
        .into_iter()
        .flat_map(<[f32; 4]>::from)
        .collect::<Vec<_>>();
    let image = Rgba32FImage::from_raw(
        data.screen_domain.width(),
        data.screen_domain.height(),
        image,
    )
    .unwrap();
    let image: RgbaImage = image.convert();

    if input.just_pressed(KeyCode::KeyE) {
        image.save("output.png").unwrap();
    }

    let image = images.add(Image::from_dynamic(
        image.into(),
        false,
        RenderAssetUsages::RENDER_WORLD,
    ));
    *sprite.single_mut() = image;
}

#[tracked]
fn intersect(
    ray_start: Expr<Vec3>,
    ray_dir: Expr<Vec3>,
    slice: Expr<Slice>,
) -> (Expr<f32>, Expr<bool>) {
    let proj = ray_dir.dot(slice.normal);
    let t = (slice.position - ray_start).dot(slice.normal) / proj;
    (t, proj <= 0.0)
}

#[tracked]
fn intersect_sphere(
    ray_start: Expr<Vec3>,
    ray_dir: Expr<Vec3>,
    radius: Expr<f32>,
) -> (Expr<f32>, Expr<f32>, Expr<bool>) {
    let dist_to_parallel = -ray_start.dot(ray_dir);
    let min_point = ray_start + dist_to_parallel * ray_dir;
    let dist_to_center = min_point.length();
    if dist_to_center > radius {
        (0.0.expr(), 0.0.expr(), false.expr())
    } else {
        let dist_to_intersection = (radius.sqr() - dist_to_center.sqr()).sqrt();
        let min_t = dist_to_parallel - dist_to_intersection;
        let max_t = dist_to_parallel + dist_to_intersection;
        (min_t, max_t, true.expr())
    }
}

#[tracked]
fn dda(
    constants: &RenderConstants,
    ray_start: Expr<Vec3>,
    ray_dir: Expr<Vec3>,
    bounds: (Expr<Vec3>, Expr<Vec3>),
    f: impl Fn(Expr<LVec3<i32>>, Expr<f32>, Expr<f32>) -> Expr<bool>,
) {
    let ray_start = ray_start / constants.grid_scale;
    let pos = ray_start.floor().cast_i32();
    let pos = pos.var();

    // Have to transform by grid_scale since it could be uneven?
    let delta_dist = (ray_dir.length() / (ray_dir + f32::EPSILON)).abs();

    let ray_step = ray_dir.signum().cast_i32();
    let side_dist =
        (ray_dir.signum() * (pos.cast_f32() - ray_start) + ray_dir.signum() * 0.5 + 0.5)
            * delta_dist;
    let side_dist = side_dist.var();

    for _i in 0_u32.expr()..constants.grid_size.element_sum().expr() {
        let mask = side_dist <= luisa::min(side_dist.yzx(), side_dist.zxy());

        let last_t = side_dist.reduce_min();

        *side_dist += mask.select(delta_dist, Vec3::splat_expr(0.0));
        *pos += mask.select(ray_step, LVec3::splat_expr(0_i32));

        let world_pos = pos.cast_f32() * constants.grid_scale;
        if (world_pos < bounds.0).any() || (world_pos > bounds.1).any() {
            break;
        }

        let next_t = side_dist.reduce_min();

        if f(
            **pos,
            next_t * constants.grid_scale,
            last_t * constants.grid_scale,
        ) {
            break;
        }
    }
}

#[kernel(init(pub))]
fn trace_kernel(
    data: Res<RenderData>,
    constants: Res<RenderConstants>,
    particles: Res<simulation::Particles>,
    bonds: Res<simulation::Bonds>,
) -> Kernel<fn(LMat3, Vec3, Vec3, Vec3)> {
    Kernel::build(&data.screen_domain, &|pixel, view, ray_start, min, max| {
        let bounds = (min, max);
        let ray_dir = view
            * Vec3::expr(
                pixel.x.cast_f32() - data.screen_domain.width() as f32 / 2.0,
                pixel.y.cast_f32() - data.screen_domain.height() as f32 / 2.0,
                1.0,
            );
        let ray_dir = ray_dir.normalize();

        let additional_slices = constants.additional_slices.expr();

        let normal = Vec3::splat(0.0).var();

        dda(
            &constants,
            ray_start,
            ray_dir,
            bounds,
            |pos, next_t, last_t| {
                let cell = grid_cell_index(pos, constants.grid_size);

                if data.grid.occupance.read(cell) {
                    let offset = data.grid.offset.read(cell);
                    let count = data.grid.count.read(cell);
                    let best_t = f32::INFINITY.var();
                    let best_normal = Vec3::splat(0.0).var();
                    for i in 0.expr()..count {
                        let index = data.grid.particles.read(offset + i);
                        let linpos = particles.linpos.read(index);
                        let ray_pos = ray_start - linpos;
                        let (min_t, max_t, hit) =
                            intersect_sphere(ray_pos, ray_dir, constants.max_radius.expr());
                        if !hit || min_t > best_t {
                            continue;
                        }
                        let normal = (ray_pos + ray_dir * min_t).normalize();
                        let min_t = min_t.var();
                        let max_t = max_t.var();

                        let angpos = particles.angpos.read(index);
                        let start_pos = data.starting_positions.read(index);

                        let normal = qapply(conjugate(angpos), normal).var();

                        let ray_pos = qapply(conjugate(angpos), ray_pos);
                        let ray_dir = qapply(conjugate(angpos), ray_dir);

                        let bond_start = particles.bond_start.read(index);
                        let bond_count = particles.bond_count.read(index);
                        for j in 0.expr()..bond_count {
                            // TODO: Use the breaking position instead of start position so it deforms.
                            let bond = bond_start + j;
                            let bond_normal =
                                qapply(bonds.rest_rotation.read(bond), Vec3::expr(0.0, 0.0, 1.0));
                            let length = bonds.length.read(bond);
                            let slice = Slice::from_comps_expr(SliceComps {
                                normal: bond_normal,
                                position: bond_normal * length / 2.0,
                            });
                            let (t, front) = intersect(ray_pos, ray_dir, slice);
                            if front {
                                if t > min_t {
                                    *normal = bond_normal;
                                    *min_t = t;
                                }
                            } else {
                                *max_t = luisa::min(max_t, t);
                            }
                            if min_t > max_t || min_t > best_t {
                                break;
                            }
                        }
                        if min_t > max_t || min_t > best_t {
                            continue;
                        }
                        let ray_pos_off = ray_pos + start_pos;
                        for sc in 0.expr()..6_u32.expr() {
                            let slice = additional_slices[sc];
                            let (t, front) = intersect(ray_pos_off, ray_dir, slice);
                            if front {
                                if t > min_t {
                                    *normal = slice.normal;
                                    *min_t = t;
                                }
                            } else {
                                *max_t = luisa::min(max_t, t);
                            }
                            if min_t > max_t || min_t > best_t {
                                break;
                            }
                        }

                        if min_t > max_t {
                            continue;
                        }
                        if min_t < best_t {
                            *best_t = min_t;
                            *best_normal = qapply(angpos, **normal);
                        }
                    }
                    if best_t > last_t && best_t < next_t {
                        *normal = best_normal;
                        true.expr()
                    } else {
                        false.expr()
                    }
                } else {
                    false.expr()
                }
            },
        );
        let color = if (normal == Vec3::splat(0.0)).all() {
            Vec4::splat_expr(0.0)
        } else {
            (Vec3::splat_expr(0.5) + normal * 0.5).extend(1.0)
        };
        data.screen
            .write(pixel.x + data.screen_domain.width() * pixel.y, color);
    })
}
