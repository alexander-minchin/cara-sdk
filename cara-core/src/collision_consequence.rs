/// Collision consequence calculations.
/// 
/// Ported from MATLAB: CollisionConsequenceNumPieces.m and NASA_SEM_RCSToSizeVec.m

/// Output of collision consequence piece calculation.
pub struct CollisionConsequence {
    pub is_catastrophic: bool,
    pub num_pieces: i64,
}

/// Calculates the expected number of resultant debris pieces.
pub fn calculate_num_pieces(
    primary_mass: f64,
    v_rel: f64,
    secondary_mass: f64,
    lc: Option<f64>,
) -> CollisionConsequence {
    let lc = lc.unwrap_or(0.05);
    let catastrophic_threshold = 40000.0; // Joules/kg

    // ODPO relative velocity kinetic energy equation.
    let collision_energy = if secondary_mass <= primary_mass {
        0.5 * secondary_mass * v_rel.powi(2) / primary_mass
    } else {
        0.5 * primary_mass * v_rel.powi(2) / secondary_mass
    };

    let is_catastrophic = collision_energy > catastrophic_threshold;

    let big_m = if is_catastrophic {
        secondary_mass + primary_mass
    } else {
        secondary_mass.min(primary_mass) * (v_rel / 1000.0).powi(2)
    };

    let num_pieces = (0.1 * big_m.powf(0.75) * lc.powf(-1.71)).round() as i64;

    CollisionConsequence {
        is_catastrophic,
        num_pieces,
    }
}

/// NASA Size Estimation Model (SEM) RCS to Size.
///
/// Ported from MATLAB: NASA_SEM_RCSToSizeVec.m
/// z = normalized RCS (RCS/lambda^2)
/// returns normalized size (diameter/lambda)
pub fn nasa_sem_rcs_to_size(z: f64) -> f64 {
    if z <= 0.0 { return 0.0; }

    let zmie_low = 0.001220;
    let zmie_high = 2.835;

    if z > zmie_high {
        // Optical regime
        (4.0 * z / std::f64::consts::PI).sqrt()
    } else if z < zmie_low {
        // Rayleigh regime
        (z / (2.25 * std::f64::consts::PI.powi(5))).powf(1.0 / 6.0)
    } else {
        // Mie regime - logarithmic interpolation
        let xztab = [
            (0.10997, 0.001220), (0.11685, 0.001735), (0.12444, 0.002468),
            (0.13302, 0.003511), (0.14256, 0.004993), (0.15256, 0.007102),
            (0.16220, 0.01010),  (0.17138, 0.01437),  (0.18039, 0.02044),
            (0.18982, 0.02907),  (0.20014, 0.04135),  (0.21237, 0.05881),
            (0.22902, 0.08365),  (0.25574, 0.1190),   (0.30537, 0.1692),
            (0.42028, 0.2407),   (0.56287, 0.3424),   (0.71108, 0.4870),
            (0.86714, 0.6927),   (1.0529, 0.9852),    (1.2790, 1.401),
            (1.5661, 1.993),     (1.8975, 2.835)
        ];

        let log_z = z.ln();
        
        // Find interval
        for i in 0..xztab.len() - 1 {
            let z1 = xztab[i].1;
            let z2 = xztab[i+1].1;
            if z >= z1 && z <= z2 {
                let log_z1 = z1.ln();
                let log_z2 = z2.ln();
                let log_x1 = (xztab[i].0 as f64).ln();
                let log_x2 = (xztab[i+1].0 as f64).ln();
                
                let t = (log_z - log_z1) / (log_z2 - log_z1);
                let log_x = log_x1 + t * (log_x2 - log_x1);
                return log_x.exp();
            }
        }
        xztab[0].0 // Fallback
    }
}
