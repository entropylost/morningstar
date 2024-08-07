use bevy::color::Color;
use bevy::math::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    pub position: Vec3,
    pub velocity: Vec3,
    // Also rotation but that's hidden.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    pub color: Color,
    pub particle_radius: f32,
    pub bond_strength: f32,
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
}
