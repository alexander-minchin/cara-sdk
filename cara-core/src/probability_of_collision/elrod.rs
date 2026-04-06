use crate::utils::augmented_math::gen_gc_quad;
use crate::probability_of_collision::utils::remediate_covariance_2x2;
use nalgebra::{Vector3, Matrix3};
use libm::erfc;

pub struct PcElrodOutput {
    pub pc: f64,
    pub arem: [f64; 3],
    pub is_pos_def: bool,
    pub is_remediated: bool,
}

/// Compute 2D Pc using the Elrod method (Chebyshev Gaussian Quadrature).
///
/// Ported from MATLAB: PcElrod.m
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
    
    // Project to conjunction plane (x, z)
    // MATLAB: ProjectedCov = RotatedCov(:,[1 3 9]); 
    // This corresponds to (0,0), (0,2), (2,2) in 0-indexed 3x3
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

    // Reverse Cholesky factor components
    let u = rem_out.rev_chol_cov; // [a, b, c]
    
    // Chebyshev nodes and weights
    let (n_all, _, w_all) = gen_gc_quad(chebyshev_order);
    // Only retain the 2nd half
    let n_gc = &n_all[chebyshev_order / 2..];
    let w_gc = &w_all[chebyshev_order / 2..];

    let denominator = u[0] * 2.0f64.sqrt();
    let s = hbr / u[2];
    let hbr2 = hbr.powi(2);
    let x0 = r_mag; // Miss distance

    let mut p_sum = 0.0;
    for k in 0..n_gc.len() {
        let z = n_gc[k] * s;
        let radical = (hbr2 - u[2].powi(2) * z.powi(2)).sqrt();
        
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
        // Should match Foster approx 0.0025
        assert_relative_eq!(output.pc, 0.0024845, epsilon = 1e-4);
    }
}
