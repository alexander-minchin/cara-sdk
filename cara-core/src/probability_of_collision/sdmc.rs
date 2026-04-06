use nalgebra::{Vector3, Vector6, Matrix6};
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use crate::utils::augmented_math::{cov_rem_eig_val_clip, PosDefStatus};

pub struct PcSDMCOutput {
    pub pc: f64,
    pub num_hits: usize,
    pub num_trials: usize,
}

/// Simple Dynamics Monte Carlo (SDMC) Pc calculation.
///
/// Pure Rust implementation based on Hall (2018).
/// This version assumes rectilinear motion during the encounter.
pub fn pc_sdmc(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    c1: &Matrix6<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    c2: &Matrix6<f64>,
    hbr: f64,
    num_trials: usize,
    seed: u64,
) -> PcSDMCOutput {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Remediate and factorize covariances for sampling
    let l_clip = 0.0;
    // We need to convert Matrix6 to Array2 for our utility
    use ndarray::Array2;
    let c1_arr = Array2::from_shape_vec((6, 6), c1.iter().cloned().collect()).unwrap();
    let c2_arr = Array2::from_shape_vec((6, 6), c2.iter().cloned().collect()).unwrap();
    
    let rem1 = cov_rem_eig_val_clip(&c1_arr, l_clip);
    let rem2 = cov_rem_eig_val_clip(&c2_arr, l_clip);

    // Cholesky-like decomposition for sampling: A = V * sqrt(L)
    let mut sampling_mat1 = Matrix6::zeros();
    for i in 0..6 {
        let col = rem1.v_raw.column(i) * rem1.l_rem[i].sqrt();
        sampling_mat1.set_column(i, &Vector6::from_column_slice(col.as_slice()));
    }

    let mut sampling_mat2 = Matrix6::zeros();
    for i in 0..6 {
        let col = rem2.v_raw.column(i) * rem2.l_rem[i].sqrt();
        sampling_mat2.set_column(i, &Vector6::from_column_slice(col.as_slice()));
    }

    let mut num_hits = 0;

    for _ in 0..num_trials {
        // Sample deviations
        let mut z1 = Vector6::zeros();
        let mut z2 = Vector6::zeros();
        for i in 0..6 {
            z1[i] = normal.sample(&mut rng);
            z2[i] = normal.sample(&mut rng);
        }

        let dev1 = sampling_mat1 * z1;
        let dev2 = sampling_mat2 * z2;

        let r1_sampled = r1 + dev1.fixed_view::<3, 1>(0, 0);
        let v1_sampled = v1 + dev1.fixed_view::<3, 1>(3, 0);
        let r2_sampled = r2 + dev2.fixed_view::<3, 1>(0, 0);
        let v2_sampled = v2 + dev2.fixed_view::<3, 1>(3, 0);

        // Rectilinear relative motion: dr(t) = dr0 + dv0 * t
        let dr0 = r1_sampled - r2_sampled;
        let dv0 = v1_sampled - v2_sampled;

        // Minimum distance time t_min = -(dr0 . dv0) / |dv0|^2
        let dv0_mag2 = dv0.norm_squared();
        if dv0_mag2 > 1e-12 {
            let t_min = -dr0.dot(&dv0) / dv0_mag2;
            let dr_min = dr0 + dv0 * t_min;
            if dr_min.norm() <= hbr {
                num_hits += 1;
            }
        } else {
            // Static case
            if dr0.norm() <= hbr {
                num_hits += 1;
            }
        }
    }

    PcSDMCOutput {
        pc: num_hits as f64 / num_trials as f64,
        num_hits,
        num_trials,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector3, Matrix6};
    use approx::assert_relative_eq;

    #[test]
    fn test_pc_sdmc_centered() {
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let v1 = Vector3::new(0.0, 7.5, 0.0);
        let mut c1 = Matrix6::zeros();
        c1.fixed_view_mut::<3,3>(0,0).copy_from(&nalgebra::Matrix3::from_element(0.01));
        c1[(0,0)] = 0.01; c1[(1,1)] = 0.01; c1[(2,2)] = 0.01;
        
        let r2 = Vector3::new(7000.0, 0.001, 0.0);
        let v2 = Vector3::new(0.0, 0.0, 7.5);
        let mut c2 = Matrix6::zeros();
        c2[(0,0)] = 0.01; c2[(1,1)] = 0.01; c2[(2,2)] = 0.01;
        
        let hbr = 0.01;
        
        // Large number of trials for stability
        let output = pc_sdmc(&r1, &v1, &c1, &r2, &v2, &c2, hbr, 100_000, 42);
        
        assert!(output.pc > 0.0);
        // Compare with Foster approx 0.0025. 
        // 100k trials should be accurate to roughly 1/sqrt(100k) ~ 0.003 relative error.
        assert_relative_eq!(output.pc, 0.0024845, epsilon = 1e-3);
    }
}
