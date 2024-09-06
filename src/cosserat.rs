use sefirot::prelude::*;

use crate::utils::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
pub struct CosseratPdOutputs {
    pub se_lin_force: Vec3,
    pub se_ang_force: Vec3,
    pub se_lin_grad2: Vec3,
    pub se_ang_grad2: Vec3,
    pub bt_ang_force: Vec3,
    pub bt_ang_grad2: Vec3,
}

#[tracked]
#[allow(clippy::too_many_arguments)]
pub fn compute_pd(
    bend_twist_coeff: Vec3,
    stretch_shear_coeff: Vec3,
    length: Expr<f32>,
    qrest: Expr<Vec4>,
    g: Expr<Mat4x3>,
    pdiff: Expr<Vec3>,
    [qi, qj]: [Expr<Vec4>; 2],
    qdiff: Expr<Vec4>,
) -> Expr<CosseratPdOutputs> {
    let qm = (qi + qj) / 2.0;
    let qm_norm = qm.norm();
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
    ) / length;

    let qtotal = qmul(qij, qrest);

    // Inline these.
    let qtotal_mat = qrotmat(qtotal);
    let qtotal_mat_t = qtotal_mat.transpose();

    // Can also write the force using quaternion multiplication but that's less efficient.
    let strain = length.recip() * qtotal_mat_t * pdiff - Vec3::z();
    let se_lin_force = qtotal_mat * (stretch_shear_coeff * strain);
    let se_lin_grad2 = Vec3::expr(
        qtotal_mat_t.x.dot(stretch_shear_coeff * qtotal_mat_t.x),
        qtotal_mat_t.y.dot(stretch_shear_coeff * qtotal_mat_t.y),
        qtotal_mat_t.z.dot(stretch_shear_coeff * qtotal_mat_t.z),
    ) / length;

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
    ) / (length * qm_norm * qm_norm);

    CosseratPdOutputs::from_comps_expr(CosseratPdOutputsComps {
        se_lin_force,
        se_ang_force,
        se_lin_grad2,
        se_ang_grad2,
        bt_ang_force,
        bt_ang_grad2,
    })
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
pub struct CosseratPbdOutputs {
    pub se_dual_step: Vec3,
    pub se_lin_delta: Vec3,
    pub se_ang_delta: Vec3,
    pub bt_dual_step: Vec3,
    pub bt_ang_delta: Vec3,
    // For stress computation.
    // pub se_lin_force: Vec3,
    // pub bt_ang_force: Vec3,
}

#[tracked]
#[allow(clippy::too_many_arguments)]
pub fn compute_pbd(
    bend_twist_coeff: Vec3,
    stretch_shear_coeff: Vec3,
    length: Expr<f32>,
    qrest: Expr<Vec4>,
    g: Expr<Mat4x3>,
    [mi, mj]: [Expr<f32>; 2],
    [moi, moj]: [Expr<f32>; 2],
    pdiff: Expr<Vec3>,
    [qi, qj]: [Expr<Vec4>; 2],
    qdiff: Expr<Vec4>,
) -> Expr<CosseratPbdOutputs> {
    let qm = (qi + qj) / 2.0;
    let qm_norm = qm.norm();
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
    let gradient_t = gradient.transpose();

    let h = -darboux; //  - (1.0  / bend_twist_coeff) * lambda;
    let denom = Vec3::expr(
        gradient_t.x.norm_squared(),
        gradient_t.y.norm_squared(),
        gradient_t.z.norm_squared(),
    ) / (length * length)
        * (moi.recip() + moj.recip())
        + 1.0_f32.expr() / (length * bend_twist_coeff);
    let pd = denom.recip();

    let bt_dual_step = h * pd; // * step
    let bt_ang_delta = moi.recip() * gradient_t * bt_dual_step / length;

    let qtotal = qmul(qij, qrest);

    // Inline these.
    let qtotal_mat = qrotmat(qtotal);
    let qtotal_mat_t = qtotal_mat.transpose();

    // Can also write the force using quaternion multiplication but that's less efficient.
    let strain = length.recip() * qtotal_mat_t * pdiff - Vec3::z();

    // Missed a - sign here which is fixed later.
    let lin_grad_t = qtotal_mat;

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
    let ang_grad = qtotal_mat_t * mul_mat4x3(sub_mat(a, b), g);
    let ang_grad_t = ang_grad.transpose();

    let h = -strain; //  - (1.0  / stretch_shear_coeff) * lambda;
    let denom = Vec3::expr(
        lin_grad_t.x.norm_squared(),
        lin_grad_t.y.norm_squared(),
        lin_grad_t.z.norm_squared(),
    ) / (length * length)
        * (mi.recip() + mj.recip())
        + Vec3::expr(
            ang_grad_t.x.norm_squared(),
            ang_grad_t.y.norm_squared(),
            ang_grad_t.z.norm_squared(),
        ) / (length * length * qm_norm * qm_norm)
            * (moi.recip() + moj.recip())
        + 1.0_f32.expr() / (length * stretch_shear_coeff);
    let pd = denom.recip();

    let se_dual_step = h * pd; // * step
    let se_lin_delta = -mi.recip() * lin_grad_t * se_dual_step / length;
    let se_ang_delta = moi.recip() * ang_grad_t * se_dual_step / (length * qm_norm);

    CosseratPbdOutputs::from_comps_expr(CosseratPbdOutputsComps {
        se_dual_step,
        se_lin_delta,
        se_ang_delta,
        bt_dual_step,
        bt_ang_delta,
    })
}
