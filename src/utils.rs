use super::*;

pub type Vec4 = LVec4<f32>;
pub type Vec3 = LVec3<f32>;
use luisa::lang::types::vector::Mat3;

#[tracked]
pub fn step_pos_ang(pos: Expr<Vec4>, vel: Expr<Vec3>) -> Expr<Vec4> {
    (pos + qmul_vec_r(vel, pos) / 2.0).normalize()
}

#[tracked]
pub fn conjugate(q: Expr<Vec4>) -> Expr<Vec4> {
    q * Vec3::splat_expr(-1.0).extend(1.0)
}

// Extracts the imaginary part of the p * q quaternion product
#[tracked]
pub fn qmul_imag(p: Expr<Vec4>, q: Expr<Vec4>) -> Expr<Vec3> {
    p.w * q.xyz() + q.w * p.xyz() + p.xyz().cross(q.xyz())
}

#[tracked]
pub fn qmul(p: Expr<Vec4>, q: Expr<Vec4>) -> Expr<Vec4> {
    qmul_imag(p, q).extend(p.w * q.w - p.xyz().dot(q.xyz()))
}

#[tracked]
pub fn qmul_vec_r(p: Expr<Vec3>, q: Expr<Vec4>) -> Expr<Vec4> {
    (q.w * p + p.cross(q.xyz())).extend(-p.dot(q.xyz()))
}

// Rotates a vector by a quaternion
#[tracked]
pub fn qapply(q: Expr<Vec4>, v: Expr<Vec3>) -> Expr<Vec3> {
    let t = 2.0 * q.xyz().cross(v);
    v + q.w * t + q.xyz().cross(t)
}

#[tracked]
pub fn qrotmat(q: Expr<Vec4>) -> Expr<Mat3> {
    let x2 = q.x + q.x;
    let y2 = q.y + q.y;
    let z2 = q.z + q.z;
    let xx = q.x * x2;
    let xy = q.x * y2;
    let xz = q.x * z2;
    let yy = q.y * y2;
    let yz = q.y * z2;
    let zz = q.z * z2;
    let wx = q.w * x2;
    let wy = q.w * y2;
    let wz = q.w * z2;

    Mat3::expr(
        Vec3::expr(1.0 - (yy + zz), xy + wz, xz - wy),
        Vec3::expr(xy - wz, 1.0 - (xx + zz), yz + wx),
        Vec3::expr(xz + wy, yz - wx, 1.0 - (xx + yy)),
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
pub struct Mat4x3 {
    pub a: Vec4,
    pub b: Vec4,
    pub c: Vec4,
}
impl Mat4x3 {
    pub fn expr(a: Expr<Vec4>, b: Expr<Vec4>, c: Expr<Vec4>) -> Expr<Self> {
        Self::from_comps_expr(Mat4x3Comps { a, b, c })
    }
}

#[tracked]
pub fn add_mat(s: Expr<Mat4x3>, t: Expr<Mat4x3>) -> Expr<Mat4x3> {
    Mat4x3::expr(s.a + t.a, s.b + t.b, s.c + t.c)
}

#[tracked]
pub fn sub_mat(s: Expr<Mat4x3>, t: Expr<Mat4x3>) -> Expr<Mat4x3> {
    Mat4x3::expr(s.a - t.a, s.b - t.b, s.c - t.c)
}

#[tracked]
pub fn mul_mat4x3(s: Expr<Mat4x3>, t: Expr<Mat4x3>) -> Expr<Mat3> {
    Mat3::expr(
        Vec3::expr(s.a.dot(t.a), s.b.dot(t.a), s.c.dot(t.a)),
        Vec3::expr(s.a.dot(t.b), s.b.dot(t.b), s.c.dot(t.b)),
        Vec3::expr(s.a.dot(t.c), s.b.dot(t.c), s.c.dot(t.c)),
    )
}

#[tracked]
pub fn mul_scalar(s: Expr<f32>, t: Expr<Mat4x3>) -> Expr<Mat4x3> {
    Mat4x3::expr(s * t.a, s * t.b, s * t.c)
}

#[tracked]
pub fn div_scalar(t: Expr<Mat4x3>, s: Expr<f32>) -> Expr<Mat4x3> {
    Mat4x3::expr(t.a / s, t.b / s, t.c / s)
}
