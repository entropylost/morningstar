use bevy::math::*;

fn main() {
    println!(
        "{}",
        ron::ser::to_string(&Quat::from_axis_angle(Vec3::Z, 0.5)).unwrap()
    );
}
