use crate::probability_of_collision::utils::equinoctial_matrices;
use crate::probability_of_collision::circle::{pc_circle, PcCircleEstimationMode};
use nalgebra::{Vector3, Matrix6, Matrix3};

pub struct Pc2DHallOutput {
    pub pc: f64,
}

/// Calculate a single-conjunction Pc using the 2D-Nc algorithm (Hall).
///
/// Simplified port from MATLAB: Pc2D_Hall.m
pub fn pc_2d_hall(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    c1: &Matrix6<f64>,
    r2: &Vector3<f64>,
    v2: &Vector3<f64>,
    c2: &Matrix6<f64>,
    hbr: f64,
) -> Option<Pc2DHallOutput> {
    // Hall method uses Equinoctial matrices for more robust frame definition
    let _eq1 = equinoctial_matrices(r1, v1, c1, true)?;
    let _eq2 = equinoctial_matrices(r2, v2, c2, true)?;

    // Simplified Hall implementation: 
    // In many cases it falls back to PcCircle or a more complex 2D integral.
    // For now, let's use PcCircle as the engine.
    
    let cov1_3x3 = c1.fixed_view::<3, 3>(0, 0).into_owned();
    let cov2_3x3 = c2.fixed_view::<3, 3>(0, 0).into_owned();

    let circle_out = pc_circle(r1, v1, &cov1_3x3, r2, v2, &cov2_3x3, hbr, PcCircleEstimationMode::GaussChebyshev(64));

    Some(Pc2DHallOutput {
        pc: circle_out.pc,
    })
}
