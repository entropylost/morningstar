use sefirot::prelude::*;

use crate::utils::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
pub struct CosseratOutputs {
    pub se_lin_force: Vec3,
    pub se_ang_force: Vec3,
    pub se_lin_grad2: Vec3,
    pub se_ang_grad2: Vec3,
    pub bt_ang_force: Vec3,
    pub bt_ang_grad2: Vec3,
}

#[tracked]
#[allow(clippy::too_many_arguments)]
pub fn compute(
    bend_twist_coeff: Vec3,
    stretch_shear_coeff: Vec3,
    length: Expr<f32>,
    qrest: Expr<Vec4>,
    g: Expr<Mat4x3>,
    pdiff: Expr<Vec3>,
    [qi, qj]: [Expr<Vec4>; 2],
    qdiff: Expr<Vec4>,
) -> Expr<CosseratOutputs> {
    let qm = (qi + qj) / 2.0;
    let qm_norm = qm.length();
    let qij = qm / qm_norm;
    let qijc = conjugate(qij);

    let darboux = 2.0 / length * qmul_imag(qijc, qdiff);

    let qijc_mat = {
        let q = qij;
        Mat4x3::expr(
            Vec4::expr(q.w, q.z, -q.y, -q.x),
            Vec4::expr(-q.z, q.w, q.x, -q.y),
            Vec4::expr(q.y, -q.x, q.w, -q.z),
        )
    };
    let a = {
        let q = qdiff;
        Mat4x3::expr(
            Vec4::expr(-q.w, -q.z, q.y, q.x),
            Vec4::expr(q.z, -q.w, -q.x, q.y),
            Vec4::expr(-q.y, q.x, -q.w, q.z),
        )
    };
    let b = {
        let m = Vec3::expr(
            qijc_mat.a.dot(qdiff),
            qijc_mat.b.dot(qdiff),
            qijc_mat.c.dot(qdiff),
        );
        Mat4x3::expr(m.x * qij, m.y * qij, m.z * qij)
    };

    // TODO: Can remove the 2.0 probably.
    let gradient = sub_mat(
        div_scalar(sub_mat(a, b), qm_norm),
        mul_scalar(2.0_f32.expr(), qijc_mat),
    );

    // Actually gradient / length but that's shifted to later to reduce operations.
    let gradient = mul_mat4x3(gradient, g);

    let bt_ang_force = -(gradient.transpose() * (bend_twist_coeff * darboux));

    let bt_ang_grad2 = Vec3::expr(
        gradient.x.dot(bend_twist_coeff * gradient.x),
        gradient.y.dot(bend_twist_coeff * gradient.y),
        gradient.z.dot(bend_twist_coeff * gradient.z),
    ) / (length * length);

    let qtotal = qmul(qij, qrest);

    // Inline these.
    let qtotal_mat = qrotmat(qtotal);
    let qtotal_mat_t = qtotal_mat.transpose();

    // Can also write the force using quaternion multiplication but that's less efficient.
    let strain = 1.0 / length * qtotal_mat_t * pdiff - Vec3::z();
    let se_lin_force = qtotal_mat * (stretch_shear_coeff * strain);
    let se_lin_grad2 = Vec3::expr(
        qtotal_mat_t.x.dot(stretch_shear_coeff * qtotal_mat_t.x),
        qtotal_mat_t.y.dot(stretch_shear_coeff * qtotal_mat_t.y),
        qtotal_mat_t.z.dot(stretch_shear_coeff * qtotal_mat_t.z),
    ) / (length * length);

    let qpart = qmul(pdiff.extend(0.0), qij);

    let a = {
        let q = qpart;
        let m = [
            Vec3::expr(-q.w, q.z, -q.y),
            Vec3::expr(-q.z, -q.w, q.x),
            Vec3::expr(q.y, -q.x, -q.w),
            Vec3::expr(q.x, q.y, q.z),
        ];
        let qrt = qrotmat(qij).transpose();

        Mat4x3::expr(
            Vec4::expr(
                qrt.x.dot(m[0]),
                qrt.x.dot(m[1]),
                qrt.x.dot(m[2]),
                qrt.x.dot(m[3]),
            ),
            Vec4::expr(
                qrt.y.dot(m[0]),
                qrt.y.dot(m[1]),
                qrt.y.dot(m[2]),
                qrt.y.dot(m[3]),
            ),
            Vec4::expr(
                qrt.z.dot(m[0]),
                qrt.z.dot(m[1]),
                qrt.z.dot(m[2]),
                qrt.z.dot(m[3]),
            ),
        )
    };
    let b = Mat4x3::expr(pdiff.x * qij, pdiff.y * qij, pdiff.z * qij);

    // Actually the gradient / (length * qm_norm), but that's shifted to later to reduce operations.
    let gradient = qtotal_mat_t * mul_mat4x3(sub_mat(a, b), g);

    let se_ang_force = -(gradient.transpose() * (stretch_shear_coeff * strain)) / qm_norm;

    let se_ang_grad2 = Vec3::expr(
        gradient.x.dot(stretch_shear_coeff * gradient.x),
        gradient.y.dot(stretch_shear_coeff * gradient.y),
        gradient.z.dot(stretch_shear_coeff * gradient.z),
    ) / (length * length * qm_norm * qm_norm);

    CosseratOutputs::from_comps_expr(CosseratOutputsComps {
        se_lin_force,
        se_ang_force,
        se_lin_grad2,
        se_ang_grad2,
        bt_ang_force,
        bt_ang_grad2,
    })
}
