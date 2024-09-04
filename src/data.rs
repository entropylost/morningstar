use core::f32;

use super::*;

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
#[serde(default)]
pub struct Constants {
    pub substeps: u32,
    pub dt: f32,
    pub gravity: Vec3,
    pub breaking_model: BreakingModel,
    pub grid_size: UVec3,
    pub particle_radius: f32,
    pub collision_particle_radius: f32,
    pub bond_radius: f32,
    pub young_modulus: f32,
    pub shear_modulus: f32,
    pub collision_stiffness: f32,
    pub cosserat_step: ConstraintStepModel,
    pub collision_step: ConstraintStepModel,
    pub camera_position: Vec3,
    pub camera_target: Vec3,
    pub ambient_only: bool,
}

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
pub enum BreakingModel {
    Distance {
        #[serde(default)]
        max: f32,
        #[serde(default)]
        min: f32,
        #[serde(default)]
        angle: f32,
    },
    Stress {
        normal: f32,
        shear: f32,
    },
    TotalStress {
        max_stress: f32,
        #[serde(default)]
        use_collision: bool,
    },
}

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
pub enum ConstraintStepModel {
    StartingBondCount,
    CurrentBondCount,
    CollisionCount,
    Factor(f32),
}

impl Default for Constants {
    fn default() -> Self {
        Self {
            substeps: 10,
            dt: 0.016,
            gravity: Vec3::ZERO, // Vec3::new(0.0, -0.000002, 0.0),
            breaking_model: BreakingModel::Distance {
                max: 1.001,
                min: 0.0,
                angle: 0.0,
            },
            grid_size: UVec3::splat(40),
            particle_radius: 0.5,
            collision_particle_radius: 0.49,
            bond_radius: 0.5,
            young_modulus: f32::INFINITY,
            shear_modulus: f32::INFINITY,
            collision_stiffness: f32::INFINITY,
            cosserat_step: ConstraintStepModel::Factor(1.0 / 16.0),
            collision_step: ConstraintStepModel::Factor(1.0 / 16.0),
            camera_position: Vec3::new(0.0, 0.0, 50.0),
            camera_target: Vec3::ZERO,
            ambient_only: false,
        }
    }
}
impl Constants {
    pub fn grid_scale(&self) -> f32 {
        2.0 * self.collision_particle_radius
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
    pub lightness_power: f32,
    pub lightness_multiplier: f32,
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
            lightness_power: 1.6,
            lightness_multiplier: 0.01,
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
                lightness_power: object.lightness_power,
                lightness_multiplier: object.lightness_multiplier,
                particle_start: particle_offset,
                particle_count: count as u32,
            });
        }
        let mut constants = self.constants;
        constants.dt /= constants.substeps as f32;
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
    pub lightness_power: f32,
    pub lightness_multiplier: f32,
    pub particle_start: u32,
    pub particle_count: u32,
}
