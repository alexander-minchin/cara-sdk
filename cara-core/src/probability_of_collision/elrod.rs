use crate::utils::augmented_math::gen_gc_quad;
use crate::probability_of_collision::utils::remediate_covariance_2x2;
use nalgebra::{Vector3, Matrix3};
use libm::erfc;
use wide::f64x4;

pub struct PcElrodOutput {
    pub pc: f64,
    pub arem: [f64; 3],
    pub is_pos_def: bool,
    pub is_remediated: bool,
}

/// Compute 2D Pc using the Elrod method (Chebyshev Gaussian Quadrature).
///
/// Ported from MATLAB: PcElrod.m
/// Optimized with SIMD (wide).
pub fn pc_elrod(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    cov1: &Matrix3<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    cov2: &Matrix3<f64>,
    hbr: f64,
    chebyshev_order: usize,
) -> PcElrodOutput {
    let comb_cov = cov1 + cov2;
    let r_rel = r1 - r2;
    let v_rel = v1 - v2;

    let r_mag = r_rel.norm();
    let v_mag = v_rel.norm();

    // Orbit normal
    let h_vec = r_rel.cross(&v_rel);
    let h_mag = h_vec.norm();

    // Relative encounter frame
    let y_axis = v_rel / v_mag;
    let z_axis = h_vec / h_mag;
    let x_axis = y_axis.cross(&z_axis);

    let eci2xyz = Matrix3::from_rows(&[
        x_axis.transpose(),
        y_axis.transpose(),
        z_axis.transpose(),
    ]);

    let rotated_cov = eci2xyz * comb_cov * eci2xyz.transpose();
    
    let c11 = rotated_cov[(0, 0)];
    let c21 = rotated_cov[(0, 2)];
    let c22 = rotated_cov[(2, 2)];

    // Remediate non-positive definite covariances
    let rem_out = remediate_covariance_2x2([c11, c21, c22], hbr);
    
    if !rem_out.is_pos_def {
        return PcElrodOutput {
            pc: 0.0,
            arem: rem_out.arem,
            is_pos_def: false,
            is_remediated: rem_out.is_remediated,
        };
    }

    let u = rem_out.rev_chol_cov; // [a, b, c]
    
    let (n_all, _, w_all) = gen_gc_quad(chebyshev_order);
    let n_gc = &n_all[chebyshev_order / 2..];
    let w_gc = &w_all[chebyshev_order / 2..];

    let denominator = u[0] * 2.0f64.sqrt();
    let s = hbr / u[2];
    let hbr2 = hbr.powi(2);
    let u2_2 = u[2].powi(2);
    let x0 = r_mag;

    let mut p_sum = 0.0;
    let mut i = 0;
    let n = n_gc.len();

    let denom_simd = f64x4::splat(denominator);
    let x0_simd = f64x4::splat(x0);
    let u1_simd = f64x4::splat(u[1]);
    let u2_2_simd = f64x4::splat(u2_2);
    let hbr2_simd = f64x4::splat(hbr2);
    let s_simd = f64x4::splat(s);

    while i + 4 <= n {
        let n_simd = f64x4::from([n_gc[i], n_gc[i+1], n_gc[i+2], n_gc[i+3]]);
        let w_simd = f64x4::from([w_gc[i], w_gc[i+1], w_gc[i+2], w_gc[i+3]]);
        
        let z = n_simd * s_simd;
        let radical = (hbr2_simd - u2_2_simd * z * z).sqrt();
        
        let u1z = u1_simd * z;
        let arg1 = (x0_simd - u1z - radical) / denom_simd;
        let arg2 = (x0_simd + u1z - radical) / denom_simd;
        let arg3 = (x0_simd - u1z + radical) / denom_simd;
        let arg4 = (x0_simd + u1z + radical) / denom_simd;

        let mut erfc_sum = [0.0f64; 4];
        for k in 0..4 {
            erfc_sum[k] = erfc(arg1.as_array_ref()[k]) + erfc(arg2.as_array_ref()[k]) 
                        - erfc(arg3.as_array_ref()[k]) - erfc(arg4.as_array_ref()[k]);
        }
        
        let erfc_simd = f64x4::from(erfc_sum);
        let exp_part = (z * z / -2.0).exp();
        
        let term = w_simd * exp_part * erfc_simd;
        p_sum += term.reduce_add();

        i += 4;
    }

    // Remainder loop
    for k in i..n {
        let z = n_gc[k] * s;
        let radical = (hbr2 - u2_2 * z.powi(2)).sqrt();
        
        let term1 = erfc((x0 - u[1] * z - radical) / denominator);
        let term2 = erfc((x0 + u[1] * z - radical) / denominator);
        let term3 = erfc((x0 - u[1] * z + radical) / denominator);
        let term4 = erfc((x0 + u[1] * z + radical) / denominator);
        
        p_sum += w_gc[k] * (z.powi(2) / -2.0).exp() * (term1 + term2 - term3 - term4);
    }

    PcElrodOutput {
        pc: p_sum * s,
        arem: rem_out.arem,
        is_pos_def: true,
        is_remediated: rem_out.is_remediated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector3, Matrix3};
    use approx::assert_relative_eq;

    #[test]
    fn test_pc_elrod_centered() {
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let v1 = Vector3::new(0.0, 7.5, 0.0);
        let cov1 = Matrix3::new(
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        );
        
        let r2 = Vector3::new(7000.0, 0.001, 0.0);
        let v2 = Vector3::new(0.0, 0.0, 7.5);
        let cov2 = Matrix3::new(
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        );
        
        let hbr = 0.01;
        
        let output = pc_elrod(&r1, &v1, &cov1, &r2, &v2, &cov2, hbr, 64);
        
        assert!(output.pc > 0.0);
        assert_relative_eq!(output.pc, 0.0024845, epsilon = 1e-4);
    }
}
