use crate::utils::constants::MU_EARTH;
use nalgebra::{Vector3, Vector6};

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
}
