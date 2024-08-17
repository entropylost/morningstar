use core::f32;

use super::*;

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
#[serde(default)]
pub struct Constants {
    pub substeps: u32,
    pub dt: f32,
    pub gravity: Vec3,
    pub breaking_distance: f32,
    pub min_breaking_distance: f32,
    pub breaking_angle: f32,
    pub grid_size: UVec3,
    pub grid_scale: f32,
    pub particle_radius: f32,
    pub collision_particle_radius: f32,
    pub bond_radius: f32,
    pub young_modulus: f32,
    pub shear_modulus: f32,
    pub collision_stiffness: f32,
    pub camera_position: Vec3,
    pub camera_target: Vec3,
}
#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize, Default)]
pub enum SpringModel {
    #[default]
    Linear,
    Quadratic,
    InvQuadratic,
    Impulse(#[serde(default)] ImpulseConstants),
}
#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
#[serde(default)]
pub struct ImpulseConstants {
    pub bias: f32,
    pub slop: f32,
}
impl Default for ImpulseConstants {
    fn default() -> Self {
        Self {
            bias: 0.01,
            slop: 0.01,
        }
    }
}

impl Default for Constants {
    fn default() -> Self {
        Self {
            substeps: 10,
            dt: 1.0 / 600.0,
            gravity: Vec3::ZERO, // Vec3::new(0.0, -0.000002, 0.0),
            breaking_distance: 1.02,
            min_breaking_distance: 0.0,
            breaking_angle: 0.1,
            grid_size: UVec3::splat(40),
            grid_scale: 1.0, // The particle diameter.
            particle_radius: 0.5,
            collision_particle_radius: 0.5,
            bond_radius: 0.5,
            young_modulus: 1.0,
            shear_modulus: 1.0,
            collision_stiffness: 1.0,
            camera_position: Vec3::new(0.0, 0.0, 50.0),
            camera_target: Vec3::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(default)]
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
pub struct Particles {
    pub particles: Vec<Particle>,
    pub bonds: Vec<Bond>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Object {
    pub color: Color,
    // Path to file containing Particles
    pub particles: String,
    pub velocity: Vec3,
    pub position: Vec3,
    pub angle: Quat,
    pub mass: f32,
    pub unfix: bool,
    // pub rotation: Option<Quat>,
}
impl Default for Object {
    fn default() -> Self {
        Self {
            color: Color::srgb(0.5, 0.5, 0.5),
            particles: "".to_string(),
            velocity: Vec3::ZERO,
            position: Vec3::ZERO,
            angle: Quat::IDENTITY,
            mass: 1.0,
            unfix: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LoadedParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub bond_start: u32,
    pub bond_count: u32,
    pub mass: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub constants: Constants,
}
impl Scene {
    pub fn load(self) -> LoadedScene {
        let mut particles = vec![];
        let mut bonds = vec![];
        let mut objects = vec![];
        for object in self.objects {
            let Particles {
                particles: obj_particles,
                bonds: obj_bonds,
            } = ron::de::from_reader(File::open(object.particles).unwrap()).unwrap();
            let count = obj_particles.len();
            let particle_offset = particles.len() as u32;
            let bond_offset = bonds.len() as u32;
            particles.extend(obj_particles.into_iter().map(|p| LoadedParticle {
                position: object.position + object.angle * p.position,
                velocity: object.velocity + object.angle * p.velocity,
                bond_start: p.bond_start + bond_offset,
                bond_count: p.bond_count,
                mass: if p.fixed && !object.unfix {
                    f32::INFINITY
                } else {
                    object.mass
                },
            }));
            bonds.extend(obj_bonds.into_iter().map(|mut b| {
                b.other_particle += particle_offset;
                b
            }));
            objects.push(LoadedObject {
                color: object.color,
                particle_start: particle_offset,
                particle_count: count as u32,
            });
        }
        let constants = self.constants;
        LoadedScene {
            particles,
            bonds,
            objects,
            constants,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedScene {
    pub particles: Vec<LoadedParticle>,
    pub bonds: Vec<Bond>,
    pub objects: Vec<LoadedObject>,
    pub constants: Constants,
}

#[derive(Debug, Clone)]
pub struct LoadedObject {
    pub color: Color,
    pub particle_start: u32,
    pub particle_count: u32,
}
