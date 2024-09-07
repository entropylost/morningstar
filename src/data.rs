use core::f32;
use std::path::Path;

use super::*;

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
#[serde(default)]
pub struct Constants {
    /// The amount of steps to run per frame.
    pub substeps: u32,
    /// The total simulation timestep per frame. Note that this is divided by `substeps` to get the actual timestep.
    pub dt: f32,
    /// This is not applied to particles with infinite mass.
    pub gravity: Vec3,
    pub breaking_model: BreakingModel,
    /// The size of the grid used for collision detection. Each grid cell is 2 * `collision_particle_radius` wide.
    /// This being too small will not cause crashes since the grid is wrapped, but will cause a slowdown
    /// due to non-adjacent particles being in the same cell. Being too large will waste memory.
    /// The default value is 40x40x40.
    pub grid_size: UVec3,
    /// The radius of the particles used for rendering and inertia calculations.
    pub particle_radius: f32,
    /// The radius of the particles used for collision detection.
    /// This is set smaller than the particle radius used to generate the data to prevent both the bond and contacts happening at the same time.
    pub collision_particle_radius: f32,
    /// The radius of the bond, used to calculate bond stiffness.
    pub bond_radius: f32,
    /// Young's modulus of the bond material.
    /// Since morningstar uses pbd, there's no point to making this not infinite for stiff materials.
    /// See [RBDEM](http://ren-bo.net/papers/zkr_gmod2024_supp.pdf) for what this is used for.
    pub young_modulus: f32,
    /// Shear modulus of the bond material. See above.
    pub shear_modulus: f32,
    /// Stiffness of the collision constraint. Also set infinite normally.
    pub collision_stiffness: f32,
    /// The model used to compute the step size for the cosserat (bond) constraints.
    pub cosserat_step: ConstraintStepModel,
    /// The model used to compute the step size for the collision constraints.
    pub collision_step: ConstraintStepModel,
    /// The starting camera position.
    pub camera_position: Vec3,
    /// The starting position the camera is looking at.
    pub camera_target: Vec3,
    /// Whether to disable the directional light, making the scene look 2d.
    pub ambient_only: bool,
    pub background_color: Color,
    /// The amount to shrink bondless particles by.
    pub particle_shrink: f32,
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
            background_color: Color::srgb(0.6, 0.6, 0.62),
            particle_shrink: 0.0,
        }
    }
}
impl Constants {
    pub fn grid_scale(&self) -> f32 {
        2.0 * self.collision_particle_radius
    }
}

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
pub enum BreakingModel {
    /// A distance-based model that breaks bonds when length exceeds `max` or goes below `min` (ignored if they're 0).
    /// If `angle` is non-zero, the bond will also break if the magnitude of the difference between the bond particle's quaternions exceeds `angle`.
    Distance {
        #[serde(default)]
        max: f32,
        #[serde(default)]
        min: f32,
        #[serde(default)]
        angle: f32,
    },
    /// A stress-based model that breaks bonds when the normal or shear stress exceeds the given values.
    /// See [RBDEM](http://ren-bo.net/papers/zkr_gmod2024_supp.pdf) for how this is calculated.
    /// Also note that for generating this force, these young's modulus and shear modulus parameters are used instead of the ones in [`Constants`], unless they are 0.
    /// The functionality of this model will change based on the timestep.
    Stress {
        #[serde(default)]
        young_modulus: f32,
        #[serde(default)]
        shear_modulus: f32,
        normal: f32,
        shear: f32,
    },
    /// A model that breaks all bonds connected to a particle if the total stress exceeds the given value.
    /// This is timestep independent.
    TotalStress {
        max: f32,
        #[serde(default)]
        ignore_collision: bool,
    },
}

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize)]
pub enum ConstraintStepModel {
    StartingBondCount,
    CurrentBondCount,
    CollisionCount,
    Factor(f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Object {
    /// The base color of the object.
    pub color: Color,
    /// The change in lightness of the particles based on the number of remaining bonds.
    /// Computed as `lightness_multiplier * (1.0 - bonds / max_bonds) ^ lightness_power` in Oklch space.
    pub lightness_power: f32,
    pub lightness_multiplier: f32,
    /// The alpha value of fully-broken particles.
    pub alpha_target: f32,
    /// Path to file containing a serialized [`Particles`] struct.
    pub particles: String,
    /// Additional velocity of the object.
    pub velocity: Vec3,
    pub position: Vec3,
    pub rotation: Quat,
    /// The mass of all non-fixed particles in the object. Set to `f32::INFINITY` to make all particles fixed.
    pub mass: f32,
    /// Whether to unfix all particles in the object.
    pub unfix: bool,
}
impl Default for Object {
    fn default() -> Self {
        Self {
            color: Color::srgb(0.5, 0.5, 0.5),
            lightness_power: 1.6,
            lightness_multiplier: 1.2,
            alpha_target: 1.0,
            particles: "".to_string(),
            velocity: Vec3::ZERO,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            mass: 1.0,
            unfix: false,
        }
    }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub bond_start: u32,
    pub bond_count: u32,
    pub fixed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub constants: Constants,
}
impl Scene {
    pub fn load(self, dir: impl AsRef<Path>) -> LoadedScene {
        let mut particles = vec![];
        let mut bonds = vec![];
        let mut objects = vec![];
        for object in self.objects {
            let Particles {
                particles: obj_particles,
                bonds: obj_bonds,
            } = ron::de::from_reader(File::open(dir.as_ref().join(object.particles)).unwrap())
                .unwrap();
            let count = obj_particles.len();
            let particle_offset = particles.len() as u32;
            let bond_offset = bonds.len() as u32;
            particles.extend(obj_particles.into_iter().map(|p| LoadedParticle {
                position: object.position + object.rotation * p.position,
                velocity: object.velocity + object.rotation * p.velocity,
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
                alpha_target: object.alpha_target,
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
    pub alpha_target: f32,
    pub particle_start: u32,
    pub particle_count: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LoadedParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub bond_start: u32,
    pub bond_count: u32,
    pub mass: f32,
}
