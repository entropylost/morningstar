use bevy::color::Color;
use bevy::math::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub bond_start: u32,
    pub bond_count: u32,
    pub fixed: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bond {
    pub other_particle: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    pub color: Color,
    pub particles: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub particles: Vec<Particle>,
    pub bonds: Vec<Bond>,
}
