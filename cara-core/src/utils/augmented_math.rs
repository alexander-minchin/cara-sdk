use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array2, ArrayD};

/// Positive definite status of a matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PosDefStatus {
    Npd = -1,
    Psd = 0,
    Pd = 1,
}

/// Output of covariance remediation.
pub struct CovRemOutput {
    pub l_rem: Vec<f64>,
    pub l_raw: Vec<f64>,
    pub v_raw: DMatrix<f64>,
    pub pos_def_status: PosDefStatus,
    pub clip_status: bool,
    pub a_det: f64,
    pub a_inv: DMatrix<f64>,
    pub a_rem: DMatrix<f64>,
}

/// Remediate non-positive definite covariance matrix using eigenvalue clipping.
///
/// Ported from MATLAB: CovRemEigValClip.m
pub fn cov_rem_eig_val_clip(a_raw: &Array2<f64>, l_clip: f64) -> CovRemOutput {
    assert!(l_clip >= 0.0, "Lclip cannot be negative");
    let (rows, cols) = a_raw.dim();
    assert_eq!(rows, cols, "Matrix needs to be square to represent a covariance");

    // Convert ndarray to nalgebra DMatrix
    let a_mat = DMatrix::from_iterator(rows, cols, a_raw.iter().cloned());

    // Eigen-decomposition (assuming symmetric matrix as per MATLAB comments)
    let eigen = SymmetricEigen::new(a_mat.clone());
    let l_raw_vec = eigen.eigenvalues.as_slice().to_vec();
    let v_raw = eigen.eigenvectors;

    let min_l = l_raw_vec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let pos_def_status = if min_l < 0.0 {
        PosDefStatus::Npd
    } else if min_l == 0.0 {
        PosDefStatus::Psd
    } else {
        PosDefStatus::Pd
    };

    let mut l_rem_vec = l_raw_vec.clone();
    let mut clip_status = false;
    for val in l_rem_vec.iter_mut() {
        if *val < l_clip {
            *val = l_clip;
            clip_status = true;
        }
    }

    let a_det = l_rem_vec.iter().product();
    
    // Inverse: V * diag(1/L) * V^T
    let mut l_inv_diag = DMatrix::zeros(rows, cols);
    for i in 0..rows {
        l_inv_diag[(i, i)] = 1.0 / l_rem_vec[i];
    }
    let a_inv = &v_raw * l_inv_diag * v_raw.transpose();

    // Remediated matrix: V * diag(L) * V^T
    let a_rem = if clip_status {
        let mut l_rem_diag = DMatrix::zeros(rows, cols);
        for i in 0..rows {
            l_rem_diag[(i, i)] = l_rem_vec[i];
        }
        &v_raw * l_rem_diag * v_raw.transpose()
    } else {
        a_mat
    };

    CovRemOutput {
        l_rem: l_rem_vec,
        l_raw: l_raw_vec,
        v_raw,
        pos_def_status,
        clip_status,
        a_det,
        a_inv,
        a_rem,
    }
}
use libm::{erf, erfc};

/// Calculate the difference d = erf(a) - erf(b), with improved accuracy for large values.
///
/// Ported from MATLAB: erf_dif.m
pub fn erf_dif(a: f64, b: f64) -> f64 {
    let large = 3.0;
    if a.min(b) > large {
        erfc(b) - erfc(a)
    } else if a.max(b) < -large {
        erfc(-a) - erfc(-b)
    } else {
        erf(a) - erf(b)
    }
}

/// Vectorized version of erf_dif.
///
/// Ported from MATLAB: erf_vec_dif.m
pub fn erf_vec_dif(a: &ArrayD<f64>, b: &ArrayD<f64>) -> ArrayD<f64> {
    assert_eq!(a.shape(), b.shape(), "Invalid input - unequal (a, b) dimensions");
    
    let mut d = ArrayD::from_elem(a.shape(), f64::NAN);
    let large = 3.0;

    for ((&va, &vb), vd) in a.iter().zip(b.iter()).zip(d.iter_mut()) {
        let min_ab = va.min(vb);
        let max_ab = va.max(vb);

        if min_ab > large {
            *vd = erfc(vb) - erfc(va);
        } else if max_ab < -large {
            *vd = erfc(-va) - erfc(-vb);
        } else {
            *vd = erf(va) - erf(vb);
        }
    }
    d
}

/// Generate Gauss-Chebyshev nodes and weights.
///
/// Ported from MATLAB: GenGCQuad.m
pub fn gen_gc_quad(ngc: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if ngc == 16 {
        let tab = vec![
            (-9.829730996839018e-01, 1.837495178165702e-01, 6.773407894247465e-03),
            (-9.324722294043558e-01, 3.612416661871530e-01, 1.331615550646369e-02),
            (-8.502171357296142e-01, 5.264321628773557e-01, 1.940543741387547e-02),
            (-7.390089172206590e-01, 6.736956436465573e-01, 2.483389042441458e-02),
            (-6.026346363792563e-01, 7.980172272802396e-01, 2.941665508151885e-02),
            (-4.457383557765382e-01, 8.951632913550623e-01, 3.299767083121100e-02),
            (-2.736629900720829e-01, 9.618256431728190e-01, 3.545499047709023e-02),
            (-9.226835946330189e-02, 9.957341762950346e-01, 3.670493294584622e-02),
            (9.226835946330202e-02, 9.957341762950345e-01, 3.670493294584622e-02),
            (2.736629900720828e-01, 9.618256431728190e-01, 3.545499047709023e-02),
            (4.457383557765383e-01, 8.951632913550623e-01, 3.299767083121100e-02),
            (6.026346363792564e-01, 7.980172272802395e-01, 2.941665508151885e-02),
            (7.390089172206591e-01, 6.736956436465573e-01, 2.483389042441458e-02),
            (8.502171357296142e-01, 5.264321628773557e-01, 1.940543741387547e-02),
            (9.324722294043558e-01, 3.612416661871530e-01, 1.331615550646369e-02),
            (9.829730996839018e-01, 1.837495178165702e-01, 6.773407894247465e-03),
        ];
        let x_gc = tab.iter().map(|t| t.0).collect();
        let y_gc = tab.iter().map(|t| t.1).collect();
        let w_gc = tab.iter().map(|t| t.2).collect();
        (x_gc, y_gc, w_gc)
    } else if ngc == 64 {
        let tab = vec![
            (-9.988322268323265e-01, 4.831337952550822e-02, 4.657833967754515e-04),
            (-9.953316347176486e-01, 9.651392091451513e-02, 9.304789348454766e-04),
            (-9.895063994510511e-01, 1.444890495692213e-01, 1.393001296249116e-03),
            (-9.813701261394134e-01, 1.921267173537084e-01, 1.852270238580170e-03),
            (-9.709418174260519e-01, 2.393156642875583e-01, 2.307213117943439e-03),
            (-9.582458291091662e-01, 2.859456783986892e-01, 2.756767394164220e-03),
            (-9.433118132577431e-01, 3.319078531285286e-01, 3.199883112400166e-03),
            (-9.261746489577764e-01, 3.770948416883209e-01, 3.635525355359408e-03),
            (-9.068743608505453e-01, 4.214011077725294e-01, 4.062676660397876e-03),
            (-8.854560256532098e-01, 4.647231720437687e-01, 4.480339395850453e-03),
            (-8.619696668800491e-01, 5.069598538135907e-01, 4.887538091045943e-03),
            (-8.364701380102265e-01, 5.480125073546703e-01, 5.283321714564022e-03),
            (-8.090169943749473e-01, 5.877852522924733e-01, 5.666765895413191e-03),
            (-7.796743540632223e-01, 6.261851975383138e-01, 6.036975081942061e-03),
            (-7.485107481711009e-01, 6.631226582407955e-01, 6.393084633441722e-03),
            (-7.155989607441211e-01, 6.985113652489370e-01, 6.734262839554183e-03),
            (-6.810158587867969e-01, 7.322686665977737e-01, 7.059712862770465e-03),
            (-6.448422127361704e-01, 7.643157205458485e-01, 7.368674599481501e-03),
            (-6.071625078187112e-01, 7.945776797137543e-01, 7.660426455235353e-03),
            (-5.680647467311557e-01, 8.229838658936565e-01, 7.934287030054488e-03),
            (-5.276402441061325e-01, 8.494679351215212e-01, 8.189616709876989e-03),
            (-4.859834132426061e-01, 8.739680326265179e-01, 8.425819160404840e-03),
            (-4.431915455992412e-01, 8.964269372957038e-01, 8.642342719870315e-03),
            (-3.993645835656953e-01, 9.167921953165825e-01, 8.838681687467587e-03),
            (-3.546048870425355e-01, 9.350162426854148e-01, 9.014377504440393e-03),
            (-3.090169943749473e-01, 9.510565162951535e-01, 9.169019825067272e-03),
            (-2.627073781985868e-01, 9.648755533435515e-01, 9.302247475042994e-03),
            (-2.157841967678060e-01, 9.764410788292721e-01, 9.413749295017889e-03),
            (-1.683570413470385e-01, 9.857260809316509e-01, 9.503264867324932e-03),
            (-1.205366802553229e-01, 9.927088740980540e-01, 9.570585124197263e-03),
            (-7.243480016176228e-02, 9.973731496914912e-01, 9.615552836055650e-03),
            (-2.416374523613207e-02, 9.997080140801929e-01, 9.638062978725454e-03),
            (2.416374523613242e-02, 9.997080140801929e-01, 9.638062978725454e-03),
            (7.243480016176240e-02, 9.973731496914912e-01, 9.615552836055650e-03),
            (1.205366802553232e-01, 9.927088740980540e-01, 9.570585124197263e-03),
            (1.683570413470386e-01, 9.857260809316508e-01, 9.503264867324930e-03),
            (2.157841967678064e-01, 9.764410788292720e-01, 9.413749295017888e-03),
            (2.627073781985870e-01, 9.648755533435515e-01, 9.302247475042994e-03),
            (3.090169943749475e-01, 9.510565162951535e-01, 9.169019825067272e-03),
            (3.546048870425358e-01, 9.350162426854147e-01, 9.014377504440393e-03),
            (3.993645835656957e-01, 9.167921953165823e-01, 8.838681687467584e-03),
            (4.431915455992413e-01, 8.964269372957038e-01, 8.642342719870315e-03),
            (4.859834132426062e-01, 8.739680326265179e-01, 8.425819160404840e-03),
            (5.276402441061328e-01, 8.494679351215211e-01, 8.189616709876989e-03),
            (5.680647467311559e-01, 8.229838658936564e-01, 7.934287030054487e-03),
            (6.071625078187113e-01, 7.945776797137542e-01, 7.660426455235352e-03),
            (6.448422127361706e-01, 7.643157205458483e-01, 7.368674599481499e-03),
            (6.810158587867972e-01, 7.322686665977735e-01, 7.059712862770463e-03),
            (7.155989607441211e-01, 6.985113652489370e-01, 6.734262839554183e-03),
            (7.485107481711012e-01, 6.631226582407951e-01, 6.393084633441719e-03),
            (7.796743540632225e-01, 6.261851975383136e-01, 6.036975081942058e-03),
            (8.090169943749475e-01, 5.877852522924731e-01, 5.666765895413190e-03),
            (8.364701380102267e-01, 5.480125073546700e-01, 5.283321714564020e-03),
            (8.619696668800493e-01, 5.069598538135903e-01, 4.887538091045939e-03),
            (8.854560256532099e-01, 4.647231720437685e-01, 4.480339395850450e-03),
            (9.068743608505454e-01, 4.214011077725293e-01, 4.062676660397874e-03),
            (9.261746489577766e-01, 3.770948416883203e-01, 3.635525355359402e-03),
            (9.433118132577432e-01, 3.319078531285284e-01, 3.199883112400164e-03),
            (9.582458291091662e-01, 2.859456783986892e-01, 2.756767394164220e-03),
            (9.709418174260520e-01, 2.393156642875579e-01, 2.307213117943434e-03),
            (9.813701261394134e-01, 1.921267173537084e-01, 1.852270238580170e-03),
            (9.895063994510511e-01, 1.444890495692213e-01, 1.393001296249116e-03),
            (9.953316347176486e-01, 9.651392091451513e-02, 9.304789348454766e-04),
            (9.988322268323266e-01, 4.831337952550593e-02, 4.657833967754294e-04),
        ];
        let x_gc = tab.iter().map(|t| t.0).collect();
        let y_gc = tab.iter().map(|t| t.1).collect();
        let w_gc = tab.iter().map(|t| t.2).collect();
        (x_gc, y_gc, w_gc)
    } else {
        calculate_gc_quad(ngc)
    }
}

fn calculate_gc_quad(ngc: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ngc_f = ngc as f64;
    let c_gc = std::f64::consts::PI / (ngc_f + 1.0);
    let mut x_gc = Vec::with_capacity(ngc);
    let mut y_gc = Vec::with_capacity(ngc);
    let mut w_gc = Vec::with_capacity(ngc);
    let sqrt_8pi = (8.0 * std::f64::consts::PI).sqrt();

    for i in (1..=ngc).rev() {
        let v_gc = c_gc * (i as f64);
        let x = v_gc.cos();
        let y = (1.0 - x.powi(2)).sqrt();
        let w = c_gc * y / sqrt_8pi;
        x_gc.push(x);
        y_gc.push(y);
        w_gc.push(w);
    }
    (x_gc, y_gc, w_gc)
}

/// Make a covariance matrix diagonally symmetric if required.
///
/// Ported from MATLAB: cov_make_symmetric.m
pub fn cov_make_symmetric(c: &Array2<f64>) -> Array2<f64> {
    // Check for square matrix
    let (rows, cols) = c.dim();
    assert_eq!(rows, cols, "Matrix needs to be square to make symmetric.");

    // Calculate transpose
    let ct = c.t();

    // Check existing status of diagonal symmetry
    if c == &ct {
        // Original matrix is already diagonally symmetric
        c.clone()
    } else {
        // Average out any off-diagonal asymmetries
        let mut csym = (c + &ct) / 2.0;

        // Reflect about diagonal to ensure diagonal symmetry absolutely
        // In MATLAB: Csym = triu(Csym,0)+triu(Csym,1)';
        // This is equivalent to ensuring the upper and lower triangles are exactly equal.
        for i in 0..rows {
            for j in (i + 1)..cols {
                let val = csym[[i, j]];
                csym[[j, i]] = val;
            }
        }
        csym
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_cov_make_symmetric() {
        let c = array![
            [1.0, 0.1],
            [0.2, 1.0]
        ];
        let expected = array![
            [1.0, 0.15],
            [0.15, 1.0]
        ];
        let result = cov_make_symmetric(&c);
        assert_relative_eq!(result, expected);
    }

    #[test]
    fn test_already_symmetric() {
        let c = array![
            [1.0, 0.1],
            [0.1, 1.0]
        ];
        let result = cov_make_symmetric(&c);
        assert_eq!(result, c);
    }

    #[test]
    fn test_erf_dif() {
        // Normal case
        assert_relative_eq!(erf_dif(0.5, 0.1), erf(0.5) - erf(0.1));
        // Large positive case
        assert_relative_eq!(erf_dif(4.0, 3.5), erfc(3.5) - erfc(4.0));
        // Large negative case
        assert_relative_eq!(erf_dif(-4.0, -3.5), erfc(4.0) - erfc(3.5));
    }

    #[test]
    fn test_erf_vec_dif() {
        let a = array![0.5, 4.0, -4.0].into_dyn();
        let b = array![0.1, 3.5, -3.5].into_dyn();
        let result = erf_vec_dif(&a, &b);
        
        assert_relative_eq!(result[0], erf(0.5) - erf(0.1));
        assert_relative_eq!(result[1], erfc(3.5) - erfc(4.0));
        assert_relative_eq!(result[2], erfc(4.0) - erfc(3.5));
    }

    #[test]
    fn test_gen_gc_quad() {
        let (x, y, w) = gen_gc_quad(16);
        assert_eq!(x.len(), 16);
        assert_eq!(y.len(), 16);
        assert_eq!(w.len(), 16);
        // Check first point matches table
        assert_relative_eq!(x[0], -9.829730996839018e-01);
        
        // Test general calculation
        let (x2, _y2, _w2) = gen_gc_quad(10);
        assert_eq!(x2.len(), 10);
    }

    #[test]
    fn test_cov_rem_eig_val_clip() {
        let a_raw = array![
            [1.0, 2.0],
            [2.0, 1.0]
        ];
        // Eigenvalues are 3 and -1
        let l_clip = 0.1;
        let output = cov_rem_eig_val_clip(&a_raw, l_clip);
        
        assert_eq!(output.pos_def_status, PosDefStatus::Npd);
        assert!(output.clip_status);
        assert_relative_eq!(output.l_rem.iter().fold(f64::INFINITY, |a, &b| a.min(b)), 0.1);
        assert_relative_eq!(output.a_det, 3.0 * 0.1);
    }
}
