use crate::utils::augmented_math::{erf_vec_dif, gen_gc_quad};
use crate::probability_of_collision::utils::eig2x2;
use ndarray::array;
use nalgebra::{Vector3, Matrix3, Vector2};
use wide::{f64x4, CmpLe};

#[derive(Debug, Clone, Copy)]
pub enum PcCircleEstimationMode {
    CircumscribingSquare,
    EqualAreaSquare,
    GaussChebyshev(usize),
}

pub struct PcCircleOutput {
    pub pc: f64,
    pub is_pos_def: bool,
    pub is_remediated: bool,
    pub xm: f64,
    pub zm: f64,
    pub sx: f64,
    pub sz: f64,
    pub clip_bound_set: bool,
}

/// Computes Pc by integrating over a circle on the conjunction plane.
///
/// Ported from MATLAB: PcCircle.m
/// Optimized with SIMD (wide).
pub fn pc_circle(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    cov1: &Matrix3<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    cov2: &Matrix3<f64>,
    hbr: f64,
    mode: PcCircleEstimationMode,
) -> PcCircleOutput {
    let comb_cov = cov1 + cov2;
    let r_rel = r1 - r2;
    let v_rel = v1 - v2;

    let v_mag = v_rel.norm();
    let h_vec = r_rel.cross(&v_rel);
    let h_mag = h_vec.norm();

    let y_axis = v_rel / v_mag;
    let z_axis = h_vec / h_mag;
    let x_axis = y_axis.cross(&z_axis);

    let eci2xyz = Matrix3::from_rows(&[
        x_axis.transpose(),
        y_axis.transpose(),
        z_axis.transpose(),
    ]);

    let rotated_cov = eci2xyz * comb_cov * eci2xyz.transpose();
    
    let a = rotated_cov[(0, 0)];
    let b = rotated_cov[(0, 2)];
    let d = rotated_cov[(2, 2)];

    let eig = eig2x2(a, b, d);
    
    let mut l1 = eig.l1;
    let mut l2 = eig.l2;
    let v1_vec = eig.v1;
    let v2_vec = eig.v2;

    let f_clip = 1e-4;
    let l_rem = (f_clip * hbr).powi(2);
    let mut is_remediated = false;
    if l1 < l_rem {
        l1 = l_rem;
        is_remediated = true;
    }
    if l2 < l_rem {
        l2 = l_rem;
        is_remediated = true;
    }

    let is_pos_def = l2 > 0.0;
    let sx = l1.sqrt();
    let sz = l2.sqrt();

    let r_xyz = eci2xyz * r_rel;
    let r_cp = Vector2::new(r_xyz[0], r_xyz[2]);
    
    let xm = r_cp.dot(&v1_vec).abs();
    let zm = r_cp.dot(&v2_vec).abs();

    let mut clip_bound_set = false;
    let pc = match mode {
        PcCircleEstimationMode::EqualAreaSquare | PcCircleEstimationMode::CircumscribingSquare => {
            let hsq = if matches!(mode, PcCircleEstimationMode::EqualAreaSquare) {
                (std::f64::consts::PI / 4.0).sqrt() * hbr
            } else {
                hbr
            };
            let sqrt2 = 2.0f64.sqrt();
            let dx = sqrt2 * sx;
            let dz = sqrt2 * sz;
            
            let ex_arr = erf_vec_dif(&array![(xm + hsq) / dx].into_dyn(), &array![(xm - hsq) / dx].into_dyn());
            let ez_arr = erf_vec_dif(&array![(zm + hsq) / dz].into_dyn(), &array![(zm - hsq) / dz].into_dyn());
            ex_arr[0] * ez_arr[0] / 4.0
        }
        PcCircleEstimationMode::GaussChebyshev(ngc) => {
            let xlo = xm - hbr;
            let xhi = xm + hbr;
            let nsx = 4.0 * sx;
            let xlo_clip = if xlo < 0.0 { 0.0 } else { xlo };
            clip_bound_set = !((xlo > -nsx) && (xhi < xlo_clip + nsx));
            
            let (x_gc, y_gc, w_gc) = gen_gc_quad(ngc);
            let mut p_sum = 0.0;
            let sqrt2 = 2.0f64.sqrt();
            let dx = sqrt2 * sx;
            let dz = sqrt2 * sz;

            let mut i = 0;
            let dx_simd = f64x4::splat(dx);
            let dz_simd = f64x4::splat(dz);
            let xm_simd = f64x4::splat(xm);
            let zm_simd = f64x4::splat(zm);
            let hbr_simd = f64x4::splat(hbr);

            while i + 4 <= ngc {
                let x_simd = xm_simd + hbr_simd * f64x4::from([x_gc[i], x_gc[i+1], x_gc[i+2], x_gc[i+3]]);
                let hxi_simd = hbr_simd * f64x4::from([y_gc[i], y_gc[i+1], y_gc[i+2], y_gc[i+3]]);
                let w_simd = f64x4::from([w_gc[i], w_gc[i+1], w_gc[i+2], w_gc[i+3]]);

                let x_scaled = x_simd / dx_simd;
                let exp_part = (-(x_scaled * x_scaled)).exp();
                
                // erf_vec_dif call - we need a SIMD version of erf or call serial
                // For now, let's call erf_vec_dif serially for these 4 points to keep it simple
                // or implement a quick SIMD-friendly approximation.
                // Let's stick to the high-precision erf_vec_dif but called in a loop.
                let z_plus_h = zm_simd + hxi_simd;
                let z_minus_h = zm_simd - hxi_simd;
                
                let mut erf_vals = [0.0f64; 4];
                for k in 0..4 {
                    let a = z_plus_h.as_array_ref()[k] / dz;
                    let b = z_minus_h.as_array_ref()[k] / dz;
                    // Use the core implementation
                    erf_vals[k] = crate::utils::augmented_math::erf_dif(a, b);
                }
                
                let erf_simd = f64x4::from(erf_vals);
                let term = w_simd * exp_part * erf_simd;
                p_sum += term.reduce_add();

                i += 4;
            }

            // Remainder loop
            for j in i..ngc {
                let xi = xm + hbr * x_gc[j];
                let hxi = hbr * y_gc[j];
                let f_int = (-(xi / dx).powi(2)).exp() * 
                    crate::utils::augmented_math::erf_dif((zm + hxi) / dz, (zm - hxi) / dz);
                p_sum += w_gc[j] * f_int;
            }
            
            (hbr / sx) * p_sum
        }
    };

    PcCircleOutput {
        pc: pc.min(1.0),
        is_pos_def,
        is_remediated,
        xm,
        zm,
        sx,
        sz,
        clip_bound_set,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector3, Matrix3};
    use approx::assert_relative_eq;

    #[test]
    fn test_pc_circle_centered() {
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
        let output = pc_circle(&r1, &v1, &cov1, &r2, &v2, &cov2, hbr, PcCircleEstimationMode::GaussChebyshev(64));
        
        assert!(output.pc > 0.0);
        assert_relative_eq!(output.pc, 0.0024845, epsilon = 1e-4);
    }
}
