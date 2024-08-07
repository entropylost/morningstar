use std::fs::File;

use bevy::prelude::*;
use bevy_flycam::{MovementSettings, PlayerPlugin};
use bevy_sefirot::luisa::LuisaPlugin;
use luisa_compute::DeviceType;

fn install_eyre() {
    use color_eyre::config::*;
    HookBuilder::blank()
        .capture_span_trace_by_default(true)
        .add_frame_filter(Box::new(|frames| {
            let allowed = &["sefirot", "fracture"];
            frames.retain(|frame| {
                allowed.iter().any(|f| {
                    let name = if let Some(name) = frame.name.as_ref() {
                        name.as_str()
                    } else {
                        return false;
                    };

                    name.starts_with(f)
                })
            });
        }))
        .install()
        .unwrap();
}

fn main() {
    install_eyre();

    App::new()
        .add_plugins(DefaultPlugins)
        // Potentially replace with the fancy camera controller.
        .add_plugins(PlayerPlugin)
        .insert_resource(ClearColor(Color::srgb(0.7, 0.7, 0.72)))
        .insert_resource(MovementSettings {
            sensitivity: 0.00015,
            speed: 30.0,
        })
        .add_plugins(LuisaPlugin {
            device: DeviceType::Cuda,
            ..default()
        })
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let scene =
        ron::de::from_reader::<_, fracture::Scene>(File::open("scene.ron").unwrap()).unwrap();

    let mesh = meshes.add(Sphere::new(1.0));

    for object in scene.objects {
        let material = materials.add(StandardMaterial {
            base_color: object.color,
            ..default()
        });

        for point in object.points {
            commands.spawn(PbrBundle {
                mesh: mesh.clone(),
                material: material.clone(),
                transform: Transform::from_xyz(
                    point.position.x,
                    point.position.y,
                    point.position.z,
                ),
                ..default()
            });
        }
    }

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 10000.0,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_rotation_x(-1.0)),
        ..default()
    });
}
