use nalgebra::{Vector3, Vector6, Matrix6};
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use crate::utils::augmented_math::{cov_rem_eig_val_clip};
use wide::{f64x4, CmpLe};

pub struct PcSDMCOutput {
    pub pc: f64,
    pub num_hits: usize,
    pub num_trials: usize,
}

/// Simple Dynamics Monte Carlo (SDMC) Pc calculation.
///
/// Parallelized implementation using rayon and SIMD (wide).
/// Optimized with pre-generated samples if possible, or batch sampling.
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
    let l_clip = 0.0;
    use ndarray::Array2;
    let c1_arr = Array2::from_shape_vec((6, 6), c1.iter().cloned().collect()).unwrap();
    let c2_arr = Array2::from_shape_vec((6, 6), c2.iter().cloned().collect()).unwrap();
    
    let rem1 = cov_rem_eig_val_clip(&c1_arr, l_clip);
    let rem2 = cov_rem_eig_val_clip(&c2_arr, l_clip);

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

    let num_cpus = rayon::current_num_threads();
    let trials_per_cpu = num_trials / num_cpus;
    let extra_trials = num_trials % num_cpus;

    let total_hits: usize = (0..num_cpus).into_par_iter().map(|cpu_idx| {
        let n = if cpu_idx == num_cpus - 1 {
            trials_per_cpu + extra_trials
        } else {
            trials_per_cpu
        };
        
        let mut local_rng = StdRng::seed_from_u64(seed + cpu_idx as u64);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let hbr2 = f64x4::splat(hbr.powi(2));
        
        let mut local_hits = 0;
        let mut i = 0;

        // Sampling 12 values per trial (6 for pri, 6 for sec)
        // For 4 trials, that's 48 values.
        while i + 4 <= n {
            let mut samples = [0.0f64; 48];
            for k in 0..48 {
                samples[k] = normal.sample(&mut local_rng);
            }

            let mut z1 = [f64x4::ZERO; 6];
            let mut z2 = [f64x4::ZERO; 6];
            
            for j in 0..6 {
                z1[j] = f64x4::from([samples[j*4], samples[j*4+1], samples[j*4+2], samples[j*4+3]]);
                z2[j] = f64x4::from([samples[24+j*4], samples[24+j*4+1], samples[24+j*4+2], samples[24+j*4+3]]);
            }

            let mut dev1 = [f64x4::ZERO; 6];
            let mut dev2 = [f64x4::ZERO; 6];
            for row in 0..6 {
                for col in 0..6 {
                    dev1[row] += f64x4::splat(sampling_mat1[(row, col)]) * z1[col];
                    dev2[row] += f64x4::splat(sampling_mat2[(row, col)]) * z2[col];
                }
            }

            let dr0 = [
                f64x4::splat(r1[0] - r2[0]) + (dev1[0] - dev2[0]),
                f64x4::splat(r1[1] - r2[1]) + (dev1[1] - dev2[1]),
                f64x4::splat(r1[2] - r2[2]) + (dev1[2] - dev2[2]),
            ];
            let dv0 = [
                f64x4::splat(v1[0] - v2[0]) + (dev1[3] - dev2[3]),
                f64x4::splat(v1[1] - v2[1]) + (dev1[4] - dev2[4]),
                f64x4::splat(v1[2] - v2[2]) + (dev1[5] - dev2[5]),
            ];

            let dv0_mag2 = dv0[0] * dv0[0] + dv0[1] * dv0[1] + dv0[2] * dv0[2];
            let dot_product = dr0[0] * dv0[0] + dr0[1] * dv0[1] + dr0[2] * dv0[2];
            
            let t_min = -dot_product / dv0_mag2;
            let dr_min = [
                dr0[0] + dv0[0] * t_min,
                dr0[1] + dv0[1] * t_min,
                dr0[2] + dv0[2] * t_min,
            ];
            
            let dist2 = dr_min[0] * dr_min[0] + dr_min[1] * dr_min[1] + dr_min[2] * dr_min[2];
            let hits = dist2.cmp_le(hbr2);
            local_hits += hits.move_mask().count_ones() as usize;

            i += 4;
        }

        for _ in i..n {
            let mut z1 = Vector6::zeros();
            let mut z2 = Vector6::zeros();
            for j in 0..6 {
                z1[j] = normal.sample(&mut local_rng);
                z2[j] = normal.sample(&mut local_rng);
            }
            let dev1 = sampling_mat1 * z1;
            let dev2 = sampling_mat2 * z2;
            let dr0 = (r1 + dev1.fixed_view::<3, 1>(0, 0)) - (r2 + dev2.fixed_view::<3, 1>(0, 0));
            let dv0 = (v1 + dev1.fixed_view::<3, 1>(3, 0)) - (v2 + dev2.fixed_view::<3, 1>(3, 0));
            let dv0_mag2 = dv0.norm_squared();
            if dv0_mag2 > 1e-12 {
                let t_min = -dr0.dot(&dv0) / dv0_mag2;
                if (dr0 + dv0 * t_min).norm_squared() <= hbr.powi(2) {
                    local_hits += 1;
                }
            } else {
                if dr0.norm_squared() <= hbr.powi(2) {
                    local_hits += 1;
                }
            }
        }
        local_hits
    }).sum();

    PcSDMCOutput {
        pc: total_hits as f64 / num_trials as f64,
        num_hits: total_hits,
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
        c1[(0,0)] = 0.01; c1[(1,1)] = 0.01; c1[(2,2)] = 0.01;
        
        let r2 = Vector3::new(7000.0, 0.001, 0.0);
        let v2 = Vector3::new(0.0, 0.0, 7.5);
        let mut c2 = Matrix6::zeros();
        c2[(0,0)] = 0.01; c2[(1,1)] = 0.01; c2[(2,2)] = 0.01;
        
        let hbr = 0.01;
        let output = pc_sdmc(&r1, &v1, &c1, &r2, &v2, &c2, hbr, 100_000, 42);
        
        assert!(output.pc > 0.0);
        assert_relative_eq!(output.pc, 0.0024845, epsilon = 1e-3);
    }
}
