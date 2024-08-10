use core::f32;

use super::*;

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
#[serde(default)]
pub struct Constants {
    pub substeps: u32,
    pub dt: f32,
    pub gravity: Vec3,
    pub air_friction: f32,
    pub breaking_distance: f32,
    pub min_breaking_distance: f32,
    pub spring_constant: f32,
    pub damping_constant: f32,
    pub collision_constant: f32,
    pub grid_size: UVec3,
    pub grid_scale: f32,
    pub particle_radius: f32,
    pub spring_model: SpringModel,
    pub collision_model: SpringModel, // Add "Impulse" model, as well as collision radius..?
    pub floor: f32,
    pub floor_restitution: f32,
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
            air_friction: 0.0,
            breaking_distance: 1.02,
            min_breaking_distance: 0.0,
            spring_constant: 000.0,
            damping_constant: 00.0,
            collision_constant: 1000.0,
            grid_size: UVec3::splat(40),
            grid_scale: 1.0, // The particle diameter.
            particle_radius: 0.5,
            spring_model: SpringModel::Linear,
            collision_model: SpringModel::Linear,
            floor: f32::NEG_INFINITY,
            floor_restitution: 0.0,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Object {
    pub color: Color,
    // Path to file containing Particles
    pub particles: String,
    #[serde(default)]
    pub velocity: Vec3,
    #[serde(default)]
    pub position: Vec3,
    #[serde(default)]
    pub angle: Quat,
    #[serde(default)]
    pub fixed: i8,
    // pub rotation: Option<Quat>,
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
            particles.extend(obj_particles.into_iter().map(|mut p| {
                p.bond_start += bond_offset;
                p.position = object.position + object.angle * p.position;
                p.velocity = object.velocity + object.angle * p.velocity;
                if object.fixed == -1 {
                    p.fixed = false;
                } else if object.fixed == 1 {
                    p.fixed = true;
                }
                p
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
    pub particles: Vec<Particle>,
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
