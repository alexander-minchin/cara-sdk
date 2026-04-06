use crate::utils::augmented_math::cov_rem_eig_val_clip;
use ndarray::Array2;
use nalgebra::{Vector3, DMatrix};

#[derive(Debug, Clone, Copy)]
pub enum HbrType {
    Circle,
    Square,
    SquareEquArea,
}

pub struct Pc2DFosterOutput {
    pub pc: f64,
    pub a_rem: DMatrix<f64>,
    pub is_pos_def: bool,
    pub is_remediated: bool,
}

/// Compute 2D Pc according to Foster method.
///
/// Ported from MATLAB: Pc2D_Foster.m
pub fn pc_2d_foster(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    cov1: &Array2<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    cov2: &Array2<f64>,
    hbr: f64,
    rel_tol: f64,
    hbr_type: HbrType,
) -> Pc2DFosterOutput {
    // Combined position covariance
    let cov_comb = cov1.slice(ndarray::s![0..3, 0..3]).to_owned() + cov2.slice(ndarray::s![0..3, 0..3]);

    // Construct relative encounter frame
    let r_rel = r1 - r2;
    let v_rel = v1 - v2;
    let h_rel = r_rel.cross(&v_rel);

    let y_axis = v_rel.normalize();
    let z_axis = h_rel.normalize();
    let x_axis = y_axis.cross(&z_axis);

    // Transformation matrix from ECI to relative encounter plane (x, y, z)
    // In MATLAB: eci2xyz = [x; y; z];
    let eci2xyz = nalgebra::Matrix3::from_rows(&[
        x_axis.transpose(),
        y_axis.transpose(),
        z_axis.transpose(),
    ]);

    // Transform combined ECI covariance into xyz
    // We can use eci_to_ric logic here since it's a similar rotation
    let cov_comb_xyz = eci2xyz * nalgebra::Matrix3::from_iterator(cov_comb.iter().cloned()) * eci2xyz.transpose();
    
    // Projection onto xz-plane (the conjunction plane)
    // Cp = [1 0 0; 0 0 1] * covcombxyz * [1 0 0; 0 0 1]';
    let cp = Array2::from_shape_vec((2, 2), vec![
        cov_comb_xyz[(0, 0)], cov_comb_xyz[(0, 2)],
        cov_comb_xyz[(2, 0)], cov_comb_xyz[(2, 2)],
    ]).unwrap();

    // Remediate non-positive definite covariances
    let l_clip = (1e-4 * hbr).powi(2);
    let rem_out = cov_rem_eig_val_clip(&cp, l_clip);
    
    if !rem_out.l_rem.iter().all(|&l| l > 0.0) {
        panic!("Combined position covariance matrix is not positive definite when mapped to the 2-D conjunction plane.");
    }

    // Center of HBR in the relative encounter plane
    let x0 = r_rel.norm();
    let z0 = 0.0;

    // Inverse of the remediated Cp matrix
    let c_inv = rem_out.a_inv;
    let a_det = rem_out.a_det;

    // Integrand: exp(-1/2 * (x^T * C_inv * x))
    let integrand = |x: f64, z: f64| {
        let val = -0.5 * (c_inv[(0, 0)] * x.powi(2) + (c_inv[(0, 1)] + c_inv[(1, 0)]) * x * z + c_inv[(1, 1)] * z.powi(2));
        val.exp()
    };

    // Numerical integration (Simplified implementation for now)
    let pc = match hbr_type {
        HbrType::Circle => {
            integrate_circle(integrand, x0, z0, hbr, rel_tol)
        }
        HbrType::Square => {
            integrate_square(integrand, x0 - hbr, x0 + hbr, z0 - hbr, z0 + hbr, rel_tol)
        }
        HbrType::SquareEquArea => {
            let side = (std::f64::consts::PI.sqrt() / 2.0) * hbr;
            integrate_square(integrand, x0 - side, x0 + side, z0 - side, z0 + side, rel_tol)
        }
    };

    let final_pc = pc / (2.0 * std::f64::consts::PI * a_det.sqrt());

    Pc2DFosterOutput {
        pc: final_pc,
        a_rem: rem_out.a_rem,
        is_pos_def: true,
        is_remediated: rem_out.clip_status,
    }
}

fn integrate_circle<F>(f: F, x0: f64, z0: f64, r: f64, _tol: f64) -> f64 
where F: Fn(f64, f64) -> f64 {
    // Simple implementation using Gauss-Chebyshev nodes (using ported gen_gc_quad)
    use crate::utils::augmented_math::gen_gc_quad;
    let (_x_nodes, _y_nodes, _w_weights) = gen_gc_quad(64);
    
    // This is a simplified 2D integration for a circle.
    // For a real implementation, we might need a more robust approach.
    // Foster's method often uses a specific transformation.
    // For now, let's use a simple polar integration.
    let mut sum = 0.0;
    let n_theta = 64;
    let n_r = 64;
    
    for i in 0..n_theta {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n_theta as f64);
        for j in 0..n_r {
            let rho = r * ((j as f64 + 0.5) / (n_r as f64)).sqrt(); // area-weighted sampling
            let x = x0 + rho * theta.cos();
            let z = z0 + rho * theta.sin();
            sum += f(x, z);
        }
    }
    
    sum * (std::f64::consts::PI * r.powi(2)) / (n_theta as f64 * n_r as f64)
}

fn integrate_square<F>(f: F, x_min: f64, x_max: f64, z_min: f64, z_max: f64, _tol: f64) -> f64 
where F: Fn(f64, f64) -> f64 {
    let n = 64;
    let mut sum = 0.0;
    let dx = (x_max - x_min) / (n as f64);
    let dz = (z_max - z_min) / (n as f64);
    
    for i in 0..n {
        let x = x_min + (i as f64 + 0.5) * dx;
        for j in 0..n {
            let z = z_min + (j as f64 + 0.5) * dz;
            sum += f(x, z);
        }
    }
    
    sum * dx * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_pc_2d_foster_centered() {
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let v1 = Vector3::new(0.0, 7.5, 0.0);
        let cov1 = array![
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01]
        ];
        
        // Small offset in position so relative frame is well-defined
        let r2 = Vector3::new(7000.0, 0.001, 0.0);
        let v2 = Vector3::new(0.0, 0.0, 7.5); // Perpendicular velocity
        let cov2 = array![
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01]
        ];
        
        let hbr = 0.01; // 10 meters
        let rel_tol = 1e-8;
        
        let output = pc_2d_foster(&r1, &v1, &cov1, &r2, &v2, &cov2, hbr, rel_tol, HbrType::Circle);
        
        assert!(output.pc > 0.0);
        // Approximately 0.0025 as calculated before
        assert_relative_eq!(output.pc, 0.0024845, epsilon = 1e-4);
    }
}
