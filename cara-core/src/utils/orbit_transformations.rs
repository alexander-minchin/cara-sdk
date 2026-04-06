use crate::utils::augmented_math::cov_make_symmetric;
use crate::utils::constants::MU_EARTH;
use ndarray::Array2;
use nalgebra::{Matrix3, Vector3, Vector6};

/// Rotate ECI covariance matrix to RIC frame.
///
/// Ported from MATLAB: ECI2RIC.m
pub fn eci_to_ric(eci_cov: &Array2<f64>, r: &Vector3<f64>, v: &Vector3<f64>, make_symmetric: bool) -> Array2<f64> {
    let h = r.cross(v);
    let rhat = r.normalize();
    let chat = h.normalize();
    let ihat = chat.cross(&rhat);

    let eci_to_ric_3x3 = Matrix3::from_rows(&[
        rhat.transpose(),
        ihat.transpose(),
        chat.transpose(),
    ]);

    let size = eci_cov.nrows();
    let mut ric_cov: Array2<f64>;

    if size == 3 {
        let eci_mat = Matrix3::from_iterator(eci_cov.iter().cloned());
        let ric_mat = eci_to_ric_3x3 * eci_mat * eci_to_ric_3x3.transpose();
        ric_cov = Array2::from_shape_vec((3, 3), ric_mat.iter().cloned().collect()).unwrap();
    } else if size >= 6 {
        let mut eci_to_ric_full = Array2::zeros((size, size));
        
        // Fill the 6x6 part
        for i in 0..3 {
            for j in 0..3 {
                let val = eci_to_ric_3x3[(i, j)];
                eci_to_ric_full[[i, j]] = val;
                eci_to_ric_full[[i + 3, j + 3]] = val;
            }
        }
        
        // Fill the remaining diagonal with 1.0
        for i in 6..size {
            eci_to_ric_full[[i, i]] = 1.0;
        }

        // Convert to nalgebra matrix for multiplication or stay with ndarray
        // Given we used ndarray for cov_make_symmetric, let's use ndarray here.
        let temp = eci_to_ric_full.dot(eci_cov);
        ric_cov = temp.dot(&eci_to_ric_full.t());
    } else {
        panic!("Unsupported covariance matrix size: {}", size);
    }

    if make_symmetric {
        ric_cov = cov_make_symmetric(&ric_cov);
    }

    ric_cov
}

/// Convert mean anomaly to eccentric anomaly.
///
/// Ported from MATLAB: Mean2EccAnomaly.m
pub fn mean_to_ecc_anomaly(m: f64, e: f64) -> f64 {
    let mut ea = if (-std::f64::consts::PI < m && m < 0.0) || m > std::f64::consts::PI {
        m - e
    } else {
        m + e
    };

    let tol = 1e-13;
    let mut delta = 1.0;

    while delta > tol {
        let ea_new = ea + (m - ea + e * ea.sin()) / (1.0 - e * ea.cos());
        delta = (ea_new - ea).abs();
        ea = ea_new;
    }

    ea
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    True,
    Mean,
    Eccentric,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnitType {
    Deg,
    Rad,
}

/// Equinoctial elements.
#[derive(Debug, Clone, Copy)]
pub struct EquinoctialElements {
    pub a: f64,
    pub n: f64,
    pub af: f64,
    pub ag: f64,
    pub chi: f64,
    pub psi: f64,
    pub lm: f64,
    pub f: f64,
}

/// Convert Cartesian state to equinoctial elements.
///
/// Ported from MATLAB: convert_cartesian_to_equinoctial.m
pub fn cart_to_equinoctial(
    r_vec: &Vector3<f64>,
    v_vec: &Vector3<f64>,
    fr: f64,
    mu: f64,
) -> Option<EquinoctialElements> {
    assert!(fr.abs() == 1.0, "fr must be either +1 or -1");

    let r = r_vec.norm();
    let v2 = v_vec.norm_squared();

    let a = mu * r / (2.0 * mu - v2 * r);

    if a <= 1e-5 || a.is_infinite() {
        return None;
    }

    let rdv = r_vec.dot(v_vec);
    let rcv = r_vec.cross(v_vec);

    let n = (mu / a.powi(3)).sqrt();

    let e_vec = ((v2 - mu / r) * r_vec - rdv * v_vec) / mu;
    let ec2 = e_vec.norm_squared();

    if ec2 >= 1.0 {
        return None;
    }

    let what = rcv.normalize();

    if what[2] + fr <= 1e-10 {
        return None;
    }

    let cpden = 1.0 + fr * what[2];
    let chi = what[0] / cpden;
    let psi = -what[1] / cpden;

    let chi2 = chi.powi(2);
    let psi2 = psi.powi(2);
    let c = 1.0 + chi2 + psi2;

    let fhat = Vector3::new(1.0 - chi2 + psi2, 2.0 * chi * psi, -2.0 * fr * chi) / c;
    let ghat = Vector3::new(2.0 * fr * chi * psi, (1.0 + chi2 - psi2) * fr, 2.0 * psi) / c;

    let af = fhat.dot(&e_vec);
    let ag = ghat.dot(&e_vec);

    let af2 = af.powi(2);
    let ag2 = ag.powi(2);
    let ec2_check = ag2 + af2;

    if ec2_check >= 1.0 {
        return None;
    }

    let x = fhat.dot(r_vec);
    let y = ghat.dot(r_vec);

    let safg = (1.0 - ag2 - af2).sqrt();
    let b = 1.0 / (1.0 + safg);
    let f_den = a * safg;

    let bagaf = b * ag * af;

    let sin_f = ag + ((1.0 - ag2 * b) * y - bagaf * x) / f_den;
    let cos_f = af + ((1.0 - af2 * b) * x - bagaf * y) / f_den;
    let f = sin_f.atan2(cos_f);

    let lm = f + ag * f.cos() - af * f.sin();

    Some(EquinoctialElements {
        a, n, af, ag, chi, psi, lm, f,
    })
}

/// Calculate Jacobian dX/dE from equinoctial elements to Cartesian state.
///
/// Ported from MATLAB: jacobian_equinoctial_to_cartesian.m
pub fn jacobian_equinoctial_to_cartesian(
    e: &Vector6<f64>,
    x: &Vector6<f64>,
    fr: f64,
    mu: f64,
) -> nalgebra::Matrix6<f64> {
    let n = e[0];
    let af = e[1];
    let ag = e[2];
    let chi = e[3];
    let psi = e[4];

    let r_vec = Vector3::new(x[0], x[1], x[2]);
    let v_vec = Vector3::new(x[3], x[4], x[5]);

    let r2 = r_vec.norm_squared();
    let r = r2.sqrt();
    let r3 = r2 * r;

    let a3 = mu / n.powi(2);
    let a = a3.powf(1.0 / 3.0);
    let cap_a = n * a.powi(2);

    let ag2 = ag.powi(2);
    let af2 = af.powi(2);
    let cap_b = (1.0 - ag2 - af2).sqrt();

    let chi2 = chi.powi(2);
    let psi2 = psi.powi(2);
    let cap_c = 1.0 + chi2 + psi2;

    let fhat = Vector3::new(1.0 - chi2 + psi2, 2.0 * chi * psi, -2.0 * fr * chi) / cap_c;
    let ghat = Vector3::new(2.0 * fr * chi * psi, (1.0 + chi2 - psi2) * fr, 2.0 * psi) / cap_c;
    let what = Vector3::new(2.0 * chi, -2.0 * psi, (1.0 - chi2 - psi2) * fr) / cap_c;

    let cap_x = fhat.dot(&r_vec);
    let cap_y = ghat.dot(&r_vec);
    let x_dot = fhat.dot(&v_vec);
    let y_dot = ghat.dot(&v_vec);

    let ab = cap_a * cap_b;
    let bp1 = cap_b + 1.0;
    let n_bp1 = n * bp1;
    let a_r3 = cap_a / r3;

    let dx_daf = ag * x_dot / n_bp1 + a * (cap_y * x_dot / ab - 1.0);
    let dy_daf = ag * y_dot / n_bp1 - a * (cap_x * x_dot / ab);
    let dx_dag = -af * x_dot / n_bp1 + a * (cap_y * y_dot / ab);
    let dy_dag = -af * y_dot / n_bp1 - a * (cap_x * y_dot / ab + 1.0);

    let dx_ddaf = a * x_dot * y_dot / ab - a_r3 * (a * ag * cap_x / bp1 + cap_x * cap_y / cap_b);
    let dy_ddaf = -a * x_dot.powi(2) / ab - a_r3 * (a * ag * cap_y / bp1 - cap_x.powi(2) / cap_b);
    let dx_ddag = a * y_dot.powi(2) / ab + a_r3 * (a * af * cap_x / bp1 - cap_y.powi(2) / cap_b);
    let dy_ddag = -a * x_dot * y_dot / ab + a_r3 * (a * af * cap_y / bp1 + cap_x * cap_y / cap_b);

    let mut j = nalgebra::Matrix6::zeros();

    let cv = 1.0 / (3.0 * n);
    let cr = -2.0 * cv;
    j.fixed_view_mut::<3, 1>(0, 0).copy_from(&(cr * r_vec));
    j.fixed_view_mut::<3, 1>(3, 0).copy_from(&(cv * v_vec));

    j.fixed_view_mut::<3, 1>(0, 1).copy_from(&(dx_daf * fhat + dy_daf * ghat));
    j.fixed_view_mut::<3, 1>(3, 1).copy_from(&(dx_ddaf * fhat + dy_ddaf * ghat));

    j.fixed_view_mut::<3, 1>(0, 2).copy_from(&(dx_dag * fhat + dy_dag * ghat));
    j.fixed_view_mut::<3, 1>(3, 2).copy_from(&(dx_ddag * fhat + dy_ddag * ghat));

    let cc = 2.0 / cap_c;
    j.fixed_view_mut::<3, 1>(0, 3).copy_from(&(cc * (fr * psi * (cap_y * fhat - cap_x * ghat) - cap_x * what)));
    j.fixed_view_mut::<3, 1>(3, 3).copy_from(&(cc * (fr * psi * (y_dot * fhat - x_dot * ghat) - x_dot * what)));

    j.fixed_view_mut::<3, 1>(0, 4).copy_from(&(cc * (fr * chi * (cap_x * ghat - cap_y * fhat) + cap_y * what)));
    j.fixed_view_mut::<3, 1>(3, 4).copy_from(&(cc * (fr * chi * (x_dot * ghat - y_dot * fhat) + y_dot * what)));

    j.fixed_view_mut::<3, 1>(0, 5).copy_from(&(v_vec / n));
    j.fixed_view_mut::<3, 1>(3, 5).copy_from(&((-n * a3 / r3) * r_vec));

    j
}

/// Keplerian elements.
#[derive(Debug, Clone, Copy)]
pub struct KeplerianElements {
    pub a: f64,
    pub e: f64,
    pub i: f64,
    pub omega: f64,
    pub w: f64,
    pub anom: f64,
}

/// Convert Cartesian elements to Keplerian elements.
///
/// Ported from MATLAB: Cart2Kep.m
pub fn cart_to_kep(cart: &Vector6<f64>, anom_type: AnomalyType, unit_type: UnitType) -> KeplerianElements {
    let r_vec = Vector3::new(cart[0], cart[1], cart[2]);
    let v_vec = Vector3::new(cart[3], cart[4], cart[5]);

    let norm_r = r_vec.norm();
    let norm_v = v_vec.norm();

    assert!(norm_v > 0.0, "Satellite cannot have zero velocity");
    assert!(norm_r > 6378.137, "Satellite cannot be inside the Earth");

    let h_vec = r_vec.cross(&v_vec);
    let norm_h = h_vec.norm();

    assert!(norm_h > 1e-5, "Rectilinear orbit, unable to calculate Keplerian elements");

    let k_vec = Vector3::new(0.0, 0.0, 1.0);
    let n_vec = k_vec.cross(&h_vec);
    let norm_n = n_vec.norm();

    let e_vec = (v_vec.cross(&h_vec) / MU_EARTH) - (r_vec / norm_r);
    let e = e_vec.norm();

    assert!((1.0 - e).abs() > 1e-5, "Parabolic orbit, unable to calculate Keplerian elements");
    assert!(e < 1.0, "Hyperbolic orbit, unable to calculate Keplerian elements");

    let energy = 0.5 * norm_v.powi(2) - MU_EARTH / norm_r;
    let a = -0.5 * MU_EARTH / energy;

    let mut i = (h_vec[2] / norm_h).acos().to_degrees();
    let mut omega;
    let mut w;
    let mut anom;

    if e <= 1e-5 && norm_n <= 1e-5 {
        // Equatorial circular
        omega = 0.0;
        w = 0.0;
        anom = r_vec[0].atan2(r_vec[1]).to_degrees(); // This might be wrong, MATLAB uses acos and quadrant check
        // Re-implementing MATLAB's logic for consistency
        anom = (r_vec[0] / norm_r).acos().to_degrees();
        if r_vec[1] < 0.0 {
            anom = 360.0 - anom;
        }
    } else if e <= 1e-5 {
        // Inclined circular
        omega = (n_vec[0] / norm_n).acos().to_degrees();
        if n_vec[1] < 0.0 {
            omega = 360.0 - omega;
        }
        w = 0.0;
        anom = (n_vec.dot(&r_vec) / (norm_n * norm_r)).acos().to_degrees();
        if r_vec[2] < 0.0 {
            anom = 360.0 - anom;
        }
    } else if norm_n <= 1e-5 {
        // General equatorial
        omega = 0.0;
        w = (e_vec[0] / e).acos().to_degrees();
        if e_vec[1] < 0.0 {
            w = 360.0 - w;
        }
        anom = (e_vec.dot(&r_vec) / (norm_r * e)).acos().to_degrees();
        if r_vec.dot(&v_vec) < 0.0 {
            anom = 360.0 - anom;
        }
    } else {
        omega = (n_vec[0] / norm_n).acos().to_degrees();
        if n_vec[1] < 0.0 {
            omega = 360.0 - omega;
        }
        w = (n_vec.dot(&e_vec) / (norm_n * e)).acos().to_degrees();
        if e_vec[2] < 0.0 {
            w = 360.0 - w;
        }
        anom = (e_vec.dot(&r_vec) / (norm_r * e)).acos().to_degrees();
        if r_vec.dot(&v_vec) < 0.0 {
            anom = 360.0 - anom;
        }
    }

    // Compute anomaly angle
    match anom_type {
        AnomalyType::True => {}
        AnomalyType::Eccentric => {
            anom = 2.0 * ((1.0 - e).sqrt() / (1.0 + e).sqrt() * (anom / 2.0).to_radians().tan()).atan().to_degrees();
            if anom < 0.0 {
                anom += 360.0;
            }
        }
        AnomalyType::Mean => {
            let mut ea = 2.0 * ((1.0 - e).sqrt() / (1.0 + e).sqrt() * (anom / 2.0).to_radians().tan()).atan();
            if ea < 0.0 {
                ea += 2.0 * std::f64::consts::PI;
            }
            anom = (ea - e * ea.sin()).to_degrees();
        }
    }

    // Convert to radians if requested
    if unit_type == UnitType::Rad {
        i = i.to_radians();
        omega = omega.to_radians();
        w = w.to_radians();
        anom = anom.to_radians();
    }

    KeplerianElements { a, e, i, omega, w, anom }
}

/// Convert Keplerian elements to Cartesian elements.
///
/// Ported from MATLAB: Kep2Cart.m
/// Input anomaly must be Mean Anomaly in radians.
pub fn kep_to_cart(kep: &KeplerianElements) -> Vector6<f64> {
    let a = kep.a;
    let e = kep.e;
    let i = kep.i;
    let omega = kep.omega;
    let w = kep.w;
    let m = kep.anom;

    assert!((e - 1.0).abs() > 1e-5, "Parabolic or rectilinear orbit, cannot calculate Cartesian elements");
    assert!(e < 1.0, "Hyperbolic orbit, cannot calculate Cartesian elements");

    // Semilatus rectum
    let p = a * (1.0 - e.powi(2));

    // Compute eccentric anomaly from mean anomaly
    let ea = mean_to_ecc_anomaly(m, e);

    // Compute true anomaly from eccentric anomaly
    let nu = 2.0 * ((1.0 + e).sqrt() / (1.0 - e).sqrt() * (ea / 2.0).tan()).atan();

    // Position and Velocity in Perifocal Coordinate System (PQW)
    let r_pqw = (p * nu.cos()) / (1.0 + e * nu.cos());
    let x = r_pqw;
    let y = (p * nu.sin()) / (1.0 + e * nu.cos());
    let z = 0.0;

    let vx = -(MU_EARTH / p).sqrt() * nu.sin();
    let vy = (MU_EARTH / p).sqrt() * (e + nu.cos());
    let vz = 0.0;

    let a11 = omega.cos() * w.cos() - omega.sin() * w.sin() * i.cos();
    let a12 = -omega.cos() * w.sin() - omega.sin() * w.cos() * i.cos();
    let a13 = omega.sin() * i.sin();
    let a21 = omega.sin() * w.cos() + omega.cos() * w.sin() * i.cos();
    let a22 = -omega.sin() * w.sin() + omega.cos() * w.cos() * i.cos();
    let a23 = -omega.cos() * i.sin();
    let a31 = w.sin() * i.sin();
    let a32 = w.cos() * i.sin();
    let a33 = i.cos();

    let rx = a11 * x + a12 * y + a13 * z;
    let ry = a21 * x + a22 * y + a23 * z;
    let rz = a31 * x + a32 * y + a33 * z;

    let vvx = a11 * vx + a12 * vy + a13 * vz;
    let vvy = a21 * vx + a22 * vy + a23 * vz;
    let vvz = a31 * vx + a32 * vy + a33 * vz;

    let mut cart = Vector6::new(rx, ry, rz, vvx, vvy, vvz);
    
    for val in cart.iter_mut() {
        if val.abs() <= 1e-5 {
            *val = 0.0;
        }
    }

    cart
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mean_to_ecc_anomaly() {
        let m = 0.5;
        let e = 0.1;
        let ea = mean_to_ecc_anomaly(m, e);
        // M = E - e sin E
        assert_relative_eq!(m, ea - e * ea.sin(), epsilon = 1e-12);
    }

    #[test]
    fn test_cart_kep_roundtrip() {
        let cart = Vector6::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let kep = cart_to_kep(&cart, AnomalyType::Mean, UnitType::Rad);
        let cart_back = kep_to_cart(&kep);

        assert_relative_eq!(cart[0], cart_back[0], epsilon = 1e-8);
        assert_relative_eq!(cart[1], cart_back[1], epsilon = 1e-8);
        assert_relative_eq!(cart[2], cart_back[2], epsilon = 1e-8);
        assert_relative_eq!(cart[3], cart_back[3], epsilon = 1e-8);
        assert_relative_eq!(cart[4], cart_back[4], epsilon = 1e-8);
        assert_relative_eq!(cart[5], cart_back[5], epsilon = 1e-8);
    }

    #[test]
    fn test_eci_to_ric() {
        use ndarray::array;
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.5, 0.0);
        let eci_cov = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let ric_cov = eci_to_ric(&eci_cov, &r, &v, true);
        
        // Identity matrix should remain identity under rotation
        assert_relative_eq!(ric_cov[[0,0]], 1.0);
        assert_relative_eq!(ric_cov[[1,1]], 1.0);
        assert_relative_eq!(ric_cov[[2,2]], 1.0);
        assert_relative_eq!(ric_cov[[0,1]], 0.0, epsilon = 1e-15);
    }
}
