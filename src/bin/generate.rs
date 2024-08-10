use std::fs::File;

use bevy::math::{IVec3, Vec3};
use bevy::utils::{default, HashMap};
use fracture::data::*;
use nalgebra::Vector3;
use prism::shape::*;
use prism::*;
use smallvec::SmallVec;

fn main() {
    let volume = Cuboid::new(Vector3::new(20.0, 3.0, 3.0)) + Vector3::new(20.0, 0.0, 0.0);

    let points = volume.packed_points(PackedSettings {
        particle_settings: 0.5.into(),
        max_iters: 500,
        cutoff: 0.01,
        density: 1.3,
    });
    println!("Iters: {}", points.iters);
    println!("Penetration: {}", points.max_penetration);

    let mut particles = points
        .iter()
        .map(|&p| Particle {
            position: p.into(),
            ..default()
        })
        .collect::<Vec<_>>();

    let bond_radius = 1.5;

    let mut bonds = vec![];

    let mut grid: HashMap<IVec3, SmallVec<[(u32, Vec3); 8]>> = HashMap::new();

    for (i, p) in particles.iter().enumerate() {
        let ix = (p.position / bond_radius).as_ivec3();
        grid.entry(ix).or_default().push((i as u32, p.position));
    }
    for (i, p) in particles.iter_mut().enumerate() {
        let ix = (p.position / bond_radius).as_ivec3();
        p.bond_start = bonds.len() as u32;
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let ix = ix + IVec3::new(x, y, z);
                    if let Some(neighbors) = grid.get(&ix) {
                        for &(n, pos) in neighbors {
                            if n != i as u32 && (pos - p.position).length() < bond_radius {
                                bonds.push(Bond { other_particle: n });
                                p.bond_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    for p in &mut particles {
        if p.position.x < 1.0 {
            p.fixed = true;
        }
    }

    let file = File::create("rod.pts").unwrap();

    ron::ser::to_writer(file, &Particles { particles, bonds }).unwrap();
}
