use crate::utils::augmented_math::{cov_rem_eig_val_clip, gen_gc_quad, erf_dif};
use ndarray::Array2;
use nalgebra::{Vector3, DMatrix, Matrix3};
use wide::f64x4;

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
/// Optimized with SIMD Gauss-Chebyshev quadrature.
pub fn pc_2d_foster(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    cov1: &Array2<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    cov2: &Array2<f64>,
    hbr: f64,
    _rel_tol: f64,
    hbr_type: HbrType,
) -> Pc2DFosterOutput {
    let cov_comb = cov1.slice(ndarray::s![0..3, 0..3]).to_owned() + cov2.slice(ndarray::s![0..3, 0..3]);

    let r_rel = r1 - r2;
    let v_rel = v1 - v2;
    let h_rel = r_rel.cross(&v_rel);

    let y_axis = v_rel.normalize();
    let z_axis = h_rel.normalize();
    let x_axis = y_axis.cross(&z_axis);

    let eci2xyz = Matrix3::from_rows(&[
        x_axis.transpose(),
        y_axis.transpose(),
        z_axis.transpose(),
    ]);

    let cov_comb_xyz = eci2xyz * Matrix3::from_iterator(cov_comb.iter().cloned()) * eci2xyz.transpose();
    
    let cp = Array2::from_shape_vec((2, 2), vec![
        cov_comb_xyz[(0, 0)], cov_comb_xyz[(0, 2)],
        cov_comb_xyz[(2, 0)], cov_comb_xyz[(2, 2)],
    ]).unwrap();

    let l_clip = (1e-4 * hbr).powi(2);
    let rem_out = cov_rem_eig_val_clip(&cp, l_clip);
    
    if !rem_out.l_rem.iter().all(|&l| l > 0.0) {
        panic!("Combined position covariance matrix is not positive definite when mapped to the 2-D conjunction plane.");
    }

    let r_xyz = eci2xyz * r_rel;
    let xm = r_xyz[0];
    let zm = r_xyz[2];

    let cp_2x2 = nalgebra::Matrix2::new(
        rem_out.a_rem[(0, 0)], rem_out.a_rem[(0, 1)],
        rem_out.a_rem[(1, 0)], rem_out.a_rem[(1, 1)]
    );
    let eig = nalgebra::SymmetricEigen::new(cp_2x2);
    
    let l2 = eig.eigenvalues[0]; 
    let l1 = eig.eigenvalues[1]; 
    let v2_vec = eig.eigenvectors.column(0);
    let v1_vec = eig.eigenvectors.column(1);

    let sx = l1.sqrt();
    let sz = l2.sqrt();

    let r_cp = nalgebra::Vector2::new(xm, zm);
    let x0 = r_cp.dot(&v1_vec.into_owned()).abs();
    let z0 = r_cp.dot(&v2_vec.into_owned()).abs();

    let pc = match hbr_type {
        HbrType::Circle => {
            integrate_circle_gc_simd(x0, z0, sx, sz, hbr, 64)
        }
        HbrType::Square => {
            integrate_square_erf(x0, z0, sx, sz, hbr)
        }
        HbrType::SquareEquArea => {
            let hsq = (std::f64::consts::PI / 4.0).sqrt() * hbr;
            integrate_square_erf(x0, z0, sx, sz, hsq)
        }
    };

    Pc2DFosterOutput {
        pc: pc.min(1.0),
        a_rem: rem_out.a_rem,
        is_pos_def: true,
        is_remediated: rem_out.clip_status,
    }
}

fn integrate_circle_gc_simd(x0: f64, z0: f64, sx: f64, sz: f64, hbr: f64, ngc: usize) -> f64 {
    let (x_gc, y_gc, w_gc) = gen_gc_quad(ngc);
    let mut p_sum = 0.0;
    let sqrt2 = 2.0f64.sqrt();
    let dx = sqrt2 * sx;
    let dz = sqrt2 * sz;

    let mut i = 0;
    let dx_simd = f64x4::splat(dx);
    let x0_simd = f64x4::splat(x0);
    let z0_simd = f64x4::splat(z0);
    let hbr_simd = f64x4::splat(hbr);

    while i + 4 <= ngc {
        let x_simd = x0_simd + hbr_simd * f64x4::from([x_gc[i], x_gc[i+1], x_gc[i+2], x_gc[i+3]]);
        let hxi_simd = hbr_simd * f64x4::from([y_gc[i], y_gc[i+1], y_gc[i+2], y_gc[i+3]]);
        let w_simd = f64x4::from([w_gc[i], w_gc[i+1], w_gc[i+2], w_gc[i+3]]);

        let x_scaled = x_simd / dx_simd;
        let exp_part = (-(x_scaled * x_scaled)).exp();
        
        let z_plus_h = z0_simd + hxi_simd;
        let z_minus_h = z0_simd - hxi_simd;
        
        let mut erf_vals = [0.0f64; 4];
        for k in 0..4 {
            let a = z_plus_h.as_array_ref()[k] / dz;
            let b = z_minus_h.as_array_ref()[k] / dz;
            erf_vals[k] = erf_dif(a, b);
        }
        
        let erf_simd = f64x4::from(erf_vals);
        let term = w_simd * exp_part * erf_simd;
        p_sum += term.reduce_add();

        i += 4;
    }

    for j in i..ngc {
        let xi = x0 + hbr * x_gc[j];
        let hxi = hbr * y_gc[j];
        let f_int = (-(xi / dx).powi(2)).exp() * erf_dif((z0 + hxi) / dz, (z0 - hxi) / dz);
        p_sum += w_gc[j] * f_int;
    }
    
    (hbr / sx) * p_sum
}

fn integrate_square_erf(x0: f64, z0: f64, sx: f64, sz: f64, h: f64) -> f64 {
    let sqrt2 = 2.0f64.sqrt();
    let dx = sqrt2 * sx;
    let dz = sqrt2 * sz;
    let ex = erf_dif((x0 + h) / dx, (x0 - h) / dx);
    let ez = erf_dif((z0 + h) / dz, (z0 - h) / dz);
    ex * ez / 4.0
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
        let cov1 = array![[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
        let r2 = Vector3::new(7000.0, 0.001, 0.0);
        let v2 = Vector3::new(0.0, 0.0, 7.5);
        let cov2 = array![[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
        let hbr = 0.01;
        let output = pc_2d_foster(&r1, &v1, &cov1, &r2, &v2, &cov2, hbr, 1e-8, HbrType::Circle);
        assert!(output.pc > 0.0);
        assert_relative_eq!(output.pc, 0.0024845, epsilon = 1e-4);
    }
}
