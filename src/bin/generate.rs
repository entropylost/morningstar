use std::fs::File;

use amida::shape::*;
use amida::*;
use bevy::color::Color;
use bevy::math::Vec3;
use fracture::*;
use nalgebra::Vector3;
use ron::ser::PrettyConfig;

fn main() {
    let volume = Cuboid::new(
        Vector3::new(-10.0, -10.0, -10.0),
        Vector3::new(10.0, 10.0, 10.0),
    );

    let points = volume
        .grid_points(1.0)
        .into_iter()
        .map(|p| Point {
            position: p.into(),
            velocity: Vec3::ZERO,
        })
        .collect::<Vec<_>>();

    let object = Object {
        color: Color::WHITE,
        particle_radius: 1.0,
        bond_strength: 1.0,
        points,
    };

    let scene = Scene {
        objects: vec![object],
    };

    let file = File::create("scene.ron").unwrap();

    ron::ser::to_writer_pretty(file, &scene, PrettyConfig::new().depth_limit(3)).unwrap();
}
