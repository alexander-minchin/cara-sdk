use nalgebra::{Matrix2, Vector2, Vector3};
use ndarray::Array2;

pub struct Eig2x2Output {
    pub v1: Vector2<f64>,
    pub v2: Vector2<f64>,
    pub l1: f64,
    pub l2: f64,
}

/// Eigenvalue and eigenvector solver for 2x2 symmetric matrices.
pub fn eig2x2(a: f64, b: f64, d: f64) -> Eig2x2Output {
    let mat = Matrix2::new(a, b, b, d);
    let eigen = nalgebra::SymmetricEigen::new(mat);
    
    // SymmetricEigen returns eigenvalues in increasing order.
    let l2 = eigen.eigenvalues[0];
    let l1 = eigen.eigenvalues[1];
    let v2 = eigen.eigenvectors.column(0).into_owned();
    let v1 = eigen.eigenvectors.column(1).into_owned();

    Eig2x2Output { v1, v2, l1, l2 }
}

pub struct Remediate2x2Output {
    pub arem: [f64; 3],
    pub rev_chol_cov: [f64; 3],
    pub is_pos_def: bool,
    pub is_remediated: bool,
}

/// Remediate 2x2 covariance and provide reverse Cholesky factor.
pub fn remediate_covariance_2x2(
    projected_cov: [f64; 3], // [c11, c21, c22]
    hbr: f64,
) -> Remediate2x2Output {
    let mut arem = projected_cov;
    let mut npd = true;
    let mut is_remediated = false;

    // Initial check
    let eig = eig2x2(arem[0], arem[1], arem[2]);
    let mut current_eig = eig;
    if current_eig.l2 > 0.0 {
        npd = false;
    }

    let f_clips = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2];
    
    if npd {
        is_remediated = true;
        for &f_clip in &f_clips {
            let l_clip = (f_clip * hbr).powi(2);
            
            let mut l1_rem = current_eig.l1;
            let mut l2_rem = current_eig.l2;
            if l1_rem < l_clip { l1_rem = l_clip; }
            if l2_rem < l_clip { l2_rem = l_clip; }

            // Reconstruct Arem
            let v1 = current_eig.v1;
            let v2 = current_eig.v2;
            arem[0] = v1[0].powi(2) * l1_rem + v2[0].powi(2) * l2_rem;
            arem[1] = v1[0] * v1[1] * l1_rem + v2[0] * v2[1] * l2_rem;
            arem[2] = v1[1].powi(2) * l1_rem + v2[1].powi(2) * l2_rem;

            // Re-check
            let new_eig = eig2x2(arem[0], arem[1], arem[2]);
            if new_eig.l2 > 0.0 {
                current_eig = new_eig;
                npd = false;
                break;
            }
            current_eig = new_eig;
        }
    }

    if npd {
        return Remediate2x2Output {
            arem: [f64::NAN, f64::NAN, f64::NAN],
            rev_chol_cov: [f64::NAN, f64::NAN, f64::NAN],
            is_pos_def: false,
            is_remediated,
        };
    }

    // Final RevChol
    let c_rc = arem[2].sqrt();
    let b_rc = arem[1] / c_rc;
    let a_rc = (arem[0] - b_rc.powi(2)).sqrt();

    Remediate2x2Output {
        arem,
        rev_chol_cov: [a_rc, b_rc, c_rc],
        is_pos_def: true,
        is_remediated,
    }
}

use crate::utils::orbit_transformations::{cart_to_equinoctial, jacobian_equinoctial_to_cartesian};
use crate::utils::augmented_math::{cov_make_symmetric, cov_rem_eig_val_clip};
use crate::utils::constants::MU_EARTH;
use nalgebra::{Vector6, Matrix6};

pub struct EquinoctialMatricesOutput {
    pub x: Vector6<f64>,
    pub p: Matrix6<f64>,
    pub e: Vector6<f64>,
    pub j: Matrix6<f64>,
    pub k: Matrix6<f64>,
    pub q: Matrix6<f64>,
    pub q_rem_stat: bool,
    pub q_raw: Matrix6<f64>,
    pub q_rem: Matrix6<f64>,
    pub c_rem: Matrix6<f64>,
}

pub fn equinoctial_matrices(
    r: &Vector3<f64>,
    v: &Vector3<f64>,
    c: &Matrix6<f64>,
    rem_eq_cov: bool,
) -> Option<EquinoctialMatricesOutput> {
    let x = Vector6::new(r[0] / 1e3, r[1] / 1e3, r[2] / 1e3, v[0] / 1e3, v[1] / 1e3, v[2] / 1e3);
    let p = c / 1e6;

    let r_km = Vector3::new(x[0], x[1], x[2]);
    let v_km = Vector3::new(x[3], x[4], x[5]);

    let eq = cart_to_equinoctial(&r_km, &v_km, 1.0, MU_EARTH)?;
    
    let e = Vector6::new(eq.n, eq.af, eq.ag, eq.chi, eq.psi, eq.lm % (2.0 * std::f64::consts::PI));
    
    let j = jacobian_equinoctial_to_cartesian(&e, &x, 1.0, MU_EARTH);
    
    let k = j.try_inverse()?;
    
    let q_raw_mat = k * p * k.transpose();
    let q_raw_ndarray = Array2::from_shape_vec((6, 6), q_raw_mat.iter().cloned().collect()).unwrap();
    let q_sym_ndarray = cov_make_symmetric(&q_raw_ndarray);
    let q = Matrix6::from_iterator(q_sym_ndarray.iter().cloned());
    
    let (q_final, q_rem, q_rem_stat, p_final) = if rem_eq_cov {
        let rem_out = cov_rem_eig_val_clip(&q_sym_ndarray, 0.0);
        if rem_out.clip_status {
            let q_rem_mat = Matrix6::from_iterator(rem_out.a_rem.iter().cloned());
            let p_rem_mat = j * q_rem_mat * j.transpose();
            let p_rem_ndarray = Array2::from_shape_vec((6, 6), p_rem_mat.iter().cloned().collect()).unwrap();
            let p_final_ndarray = cov_make_symmetric(&p_rem_ndarray);
            (q_rem_mat, q_rem_mat, true, Matrix6::from_iterator(p_final_ndarray.iter().cloned()))
        } else {
            (q, q, false, p)
        }
    } else {
        let rem_out = cov_rem_eig_val_clip(&q_sym_ndarray, 0.0);
        (q, q, rem_out.clip_status, p)
    };

    let c_rem = p_final * 1e6;

    Some(EquinoctialMatricesOutput {
        x,
        p: p_final,
        e,
        j,
        k,
        q: q_final,
        q_rem_stat,
        q_raw: q,
        q_rem,
        c_rem,
    })
}
