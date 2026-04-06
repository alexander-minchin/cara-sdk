use nalgebra::{Vector3, Matrix3};
use crate::probability_of_collision::circle::{pc_circle, PcCircleEstimationMode};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingType {
    Primary,
    Secondary,
    Both,
}

pub struct PcDilutionOutput {
    pub pc_one: f64,
    pub diluted: bool,
    pub pc_max: f64,
    pub sf_max: f64,
    pub pc_buffer: Vec<f64>,
    pub sf_buffer: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Evaluates collision probability (Pc) dilution and calculates the associated maximum Pc value.
/// 
/// Ported from MATLAB: PcDilution.m
pub fn pc_dilution(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    cov1: &Matrix3<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    cov2: &Matrix3<f64>,
    hbr: f64,
    scaling: ScalingType,
) -> PcDilutionOutput {
    // Default parameters matching MATLAB
    let red_fact: f64 = 1.1;
    let dilut_tol: f64 = 1e-3;
    let _const_tol: f64 = 1e-6;
    let iter_max = 100;
    let sf_init: [f64; 2] = [0.5, 2.0];
    let d_lf_levels: [f64; 3] = [2.0 * 1e-2, 2.0 * 1e-3, 2.0 * 1e-4];

    let dca = (r1 - r2).norm();
    
    // Nominal Pc (Sf = 1)
    let circle_out_one = pc_circle(r1, v1, cov1, r2, v2, cov2, hbr, PcCircleEstimationMode::GaussChebyshev(64));
    let pc_one = circle_out_one.pc;

    if pc_one.is_nan() {
        return PcDilutionOutput {
            pc_one: f64::NAN, diluted: false, pc_max: f64::NAN, sf_max: f64::NAN,
            pc_buffer: vec![], sf_buffer: vec![], converged: false, iterations: 0,
        };
    }

    // Quick robust check
    let d_sf_min: f64 = d_lf_levels[0]; 
    let sf2_delta = 1.0 - d_sf_min.powi(2);
    let (f1, f2) = match scaling {
        ScalingType::Primary => (sf2_delta, 1.0),
        ScalingType::Secondary => (1.0, sf2_delta),
        ScalingType::Both => (sf2_delta, sf2_delta),
    };
    let pc_delta = pc_circle(r1, v1, &(*cov1 * f1), r2, v2, &(*cov2 * f2), hbr, PcCircleEstimationMode::GaussChebyshev(64)).pc;
    
    if pc_delta < pc_one {
        return PcDilutionOutput {
            pc_one,
            diluted: false,
            pc_max: pc_one,
            sf_max: 1.0,
            pc_buffer: vec![pc_one],
            sf_buffer: vec![1.0],
            converged: true,
            iterations: 0,
        };
    }

    // Iterative search for PcMax
    let mut lf1 = sf_init[0].log10();
    let mut lf2 = sf_init[1].log10();
    let log10_two = 2.0f64.log10();
    
    let mut n_lf_level = 0;
    let mut d_lf = (d_lf_levels[n_lf_level]).min((lf2 - lf1) / 5.0);
    
    let mut sf_buf: Vec<f64> = Vec::new();
    let mut pc_buf: Vec<f64> = Vec::new();
    let mut iterations = 0;
    let mut converged = false;

    while iterations < iter_max {
        let mut lf = Vec::new();
        let mut curr = lf1;
        while curr <= lf2 + 1e-12 {
            lf.push(curr);
            curr += d_lf;
        }
        
        for &l in &lf {
            let sf = 10.0f64.powf(l);
            let f = sf.powi(2);
            let (f1, f2) = match scaling {
                ScalingType::Primary => (f, 1.0),
                ScalingType::Secondary => (1.0, f),
                ScalingType::Both => (f, f),
            };
            let pc = pc_circle(r1, v1, &(*cov1 * f1), r2, v2, &(*cov2 * f2), hbr, PcCircleEstimationMode::GaussChebyshev(64)).pc;
            sf_buf.push(sf);
            pc_buf.push(pc);
        }

        // Sort
        let mut pairs: Vec<_> = sf_buf.iter().cloned().zip(pc_buf.iter().cloned()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sf_buf = pairs.iter().map(|p| p.0).collect();
        pc_buf = pairs.iter().map(|p| p.1).collect();

        let pc0 = pc_buf.iter().cloned().fold(0.0, f64::max);
        let pc_red = pc0 / red_fact;

        if pc0 == 0.0 {
            lf1 -= log10_two;
            lf2 += log10_two;
        } else if pc_buf[0] > pc_red {
            lf2 = sf_buf[0].log10() - d_lf;
            lf1 = lf2 - log10_two;
        } else if *pc_buf.last().unwrap() > pc_red {
            lf1 = sf_buf.last().unwrap().log10() + d_lf;
            lf2 = lf1 + log10_two;
        } else if n_lf_level < d_lf_levels.len() - 1 {
            n_lf_level += 1;
            let imax = pc_buf.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let i1 = if imax > 0 { imax - 1 } else { 0 };
            let i2 = if imax < pc_buf.len() - 1 { imax + 1 } else { pc_buf.len() - 1 };
            lf1 = sf_buf[i1].log10();
            lf2 = sf_buf[i2].log10();
            d_lf = (d_lf_levels[n_lf_level]).min((lf2 - lf1) / 5.0);
        } else {
            converged = true;
            break;
        }

        iterations += 1;
    }

    // Final unique sort
    sf_buf.push(1.0);
    pc_buf.push(pc_one);
    let mut final_pairs: Vec<_> = sf_buf.into_iter().zip(pc_buf.into_iter()).collect();
    final_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    final_pairs.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);
    
    let (mut sf_final, mut pc_final): (Vec<f64>, Vec<f64>) = final_pairs.into_iter().unzip();

    if dca > hbr {
        for p in &mut pc_final {
            if *p > 0.5 { *p = 0.5; }
        }
    }

    let (pc_max, imax) = pc_final.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, &p)| (p, i)).unwrap();
    let sf_max = sf_final[imax];
    let diluted = sf_max < 1.0 && (pc_max - pc_one) > dilut_tol;

    PcDilutionOutput {
        pc_one,
        diluted,
        pc_max,
        sf_max,
        pc_buffer: pc_final,
        sf_buffer: sf_final,
        converged,
        iterations,
    }
}

/// Calculate maximum possible Pc for an event using the Frisbee Method.
/// 
/// Ported from MATLAB: FrisbeeMaxPc.m
pub fn frisbee_max_pc(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    cov1: &Matrix3<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    cov2: &Matrix3<f64>,
    hbr: f64,
    mode: PcCircleEstimationMode,
) -> f64 {
    // Construct relative encounter frame
    let r = r1 - r2;
    let v = v1 - v2;
    let h = r.cross(&v);

    let v_mag = v.norm();
    let h_mag = h.norm();

    let y_axis = v / v_mag;
    let z_axis = h / h_mag;
    let x_axis = y_axis.cross(&z_axis);

    let eci2xyz = Matrix3::from_rows(&[
        x_axis.transpose(),
        y_axis.transpose(),
        z_axis.transpose(),
    ]);

    // Transform uncombined covariances to xyz
    let cov1_xyz = eci2xyz * cov1 * eci2xyz.transpose();
    let cov2_xyz = eci2xyz * cov2 * eci2xyz.transpose();

    // Projection onto xz-plane
    let cp1 = nalgebra::Matrix2::new(
        cov1_xyz[(0, 0)], cov1_xyz[(0, 2)],
        cov1_xyz[(2, 0)], cov1_xyz[(2, 2)],
    );
    let cp2 = nalgebra::Matrix2::new(
        cov2_xyz[(0, 0)], cov2_xyz[(0, 2)],
        cov2_xyz[(2, 0)], cov2_xyz[(2, 2)],
    );

    // Inverses
    let c1_inv = cp1.try_inverse().unwrap_or_else(|| nalgebra::Matrix2::zeros());
    let c2_inv = cp2.try_inverse().unwrap_or_else(|| nalgebra::Matrix2::zeros());

    // Center of HBR in collision plane
    let x0 = r.norm();
    let rrel = nalgebra::Vector2::new(x0, 0.0);
    let urel = rrel / rrel.norm();

    // Determine Ka
    let ka1 = (rrel.transpose() * c2_inv * rrel).norm().sqrt();
    let ka2 = (rrel.transpose() * c1_inv * rrel).norm().sqrt();

    // Find j (1 = no primary, 2 = no secondary)
    let (ka_j, j) = if ka1 > ka2 { (ka1, 1) } else { (ka2, 2) };

    let mut final_cov1 = *cov1;
    let mut final_cov2 = *cov2;

    if ka_j > 1.0 {
        let vc = x0.powi(2) * (ka_j.powi(2) - 1.0) / ka_j.powi(2);
        let cov_new_2x2 = vc * (urel * urel.transpose());
        
        let mut cov_new_3x3 = Matrix3::zeros();
        cov_new_3x3[(0, 0)] = cov_new_2x2[(0, 0)];
        cov_new_3x3[(0, 2)] = cov_new_2x2[(0, 1)];
        cov_new_3x3[(2, 0)] = cov_new_2x2[(1, 0)];
        cov_new_3x3[(2, 2)] = cov_new_2x2[(1, 1)];

        let cov_new_eci = eci2xyz.transpose() * cov_new_3x3 * eci2xyz;

        if j == 1 {
            final_cov1 = cov_new_eci;
        } else {
            final_cov2 = cov_new_eci;
        }
    }

    let output = pc_circle(r1, v1, &final_cov1, r2, v2, &final_cov2, hbr, mode);
    output.pc
}
