use std::fs::File;

use bevy::color::Color;
use bevy::math::{IVec3, Vec3};
use bevy::utils::HashMap;
use fracture::*;
use nalgebra::Vector3;
use prism::shape::*;
use prism::*;
use smallvec::SmallVec;

fn main() {
    let volume = Cuboid::new(Vector3::new(40.0, 3.0, 0.1));

    let points = volume.grid_points(GridSettings {
        border_adjust_radius: 0.0,
        grid_size: Vector3::repeat(1.0),
        cell_size: Some(1.0),
        grid_offset: None,
    });
    // println!("Iters: {}", points.iters);
    // println!("Penetration: {}", points.max_penetration);

    let mut particles = points
        .iter()
        .map(|&p| Particle {
            position: p.into(),
            velocity: Vec3::ZERO,
            bond_start: 0,
            bond_count: 0,
            fixed: false,
        })
        .collect::<Vec<_>>();

    let bond_radius = 2.5;

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
        if p.position.x.abs() > 38.0 {
            p.fixed = true;
        }
    }

    let object = Object {
        color: Color::srgb(0.5, 0.5, 0.5),
        particles: (0..particles.len() as u32).collect(),
    };

    let scene = Scene {
        objects: vec![object],
        particles,
        bonds,
    };

    let file = File::create("scene.ron").unwrap();

    ron::ser::to_writer(file, &scene).unwrap();
}
