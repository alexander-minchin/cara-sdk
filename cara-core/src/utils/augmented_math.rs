use ndarray::{Array2, ArrayD};
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
}
