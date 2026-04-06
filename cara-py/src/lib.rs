use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use cara_core::probability_of_collision::foster::{pc_2d_foster, HbrType};
use cara_core::probability_of_collision::circle::{pc_circle, PcCircleEstimationMode};
use cara_core::utils::time_transformations::timestring_to_jd;
use cara_core::utils::orbit_transformations::{cart_to_kep, eci_to_ric, AnomalyType, UnitType};
use nalgebra::{Vector3, Vector6, Matrix3};

#[pyfunction]
fn compute_2d_foster(
    r1: PyReadonlyArray1<f64>,
    v1: PyReadonlyArray1<f64>,
    cov1: PyReadonlyArray2<f64>,
    r2: PyReadonlyArray1<f64>,
    v2: PyReadonlyArray1<f64>,
    cov2: PyReadonlyArray2<f64>,
    hbr: f64,
) -> PyResult<f64> {
    let r1_vec = Vector3::from_column_slice(r1.as_slice()?);
    let v1_vec = Vector3::from_column_slice(v1.as_slice()?);
    let r2_vec = Vector3::from_column_slice(r2.as_slice()?);
    let v2_vec = Vector3::from_column_slice(v2.as_slice()?);

    let cov1_ndarray = cov1.to_owned_array();
    let cov2_ndarray = cov2.to_owned_array();

    let output = pc_2d_foster(
        &r1_vec, &v1_vec, &cov1_ndarray,
        &r2_vec, &v2_vec, &cov2_ndarray,
        hbr, 1e-8, HbrType::Circle
    );

    Ok(output.pc)
}

#[pyfunction]
fn compute_pc_circle(
    r1: PyReadonlyArray1<f64>,
    v1: PyReadonlyArray1<f64>,
    cov1: PyReadonlyArray2<f64>,
    r2: PyReadonlyArray1<f64>,
    v2: PyReadonlyArray1<f64>,
    cov2: PyReadonlyArray2<f64>,
    hbr: f64,
) -> PyResult<f64> {
    let r1_vec = Vector3::from_column_slice(r1.as_slice()?);
    let v1_vec = Vector3::from_column_slice(v1.as_slice()?);
    let r2_vec = Vector3::from_column_slice(r2.as_slice()?);
    let v2_vec = Vector3::from_column_slice(v2.as_slice()?);

    let cov1_mat = Matrix3::from_iterator(cov1.to_owned_array().iter().cloned());
    let cov2_mat = Matrix3::from_iterator(cov2.to_owned_array().iter().cloned());

    let output = pc_circle(
        &r1_vec, &v1_vec, &cov1_mat,
        &r2_vec, &v2_vec, &cov2_mat,
        hbr, PcCircleEstimationMode::GaussChebyshev(64)
    );

    Ok(output.pc)
}

#[pyfunction]
fn rotate_eci_to_ric(
    py: Python<'_>,
    eci_cov: PyReadonlyArray2<f64>,
    r: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let r_vec = Vector3::from_column_slice(r.as_slice()?);
    let v_vec = Vector3::from_column_slice(v.as_slice()?);
    let eci_cov_ndarray = eci_cov.to_owned_array();
    
    let ric_cov = eci_to_ric(&eci_cov_ndarray, &r_vec, &v_vec, true);
    
    Ok(ric_cov.to_pyarray(py).to_object(py))
}

#[pyclass]
#[derive(Clone)]
struct KeplerianElements {
    #[pyo3(get)]
    pub a: f64,
    #[pyo3(get)]
    pub e: f64,
    #[pyo3(get)]
    pub i: f64,
    #[pyo3(get)]
    pub omega: f64,
    #[pyo3(get)]
    pub w: f64,
    #[pyo3(get)]
    pub anom: f64,
}

#[pyfunction]
fn cart_to_keplerian(cart: PyReadonlyArray1<f64>) -> PyResult<KeplerianElements> {
    let cart_vec = Vector6::from_column_slice(cart.as_slice()?);
    let kep = cart_to_kep(&cart_vec, AnomalyType::Mean, UnitType::Deg);
    
    Ok(KeplerianElements {
        a: kep.a,
        e: kep.e,
        i: kep.i,
        omega: kep.omega,
        w: kep.w,
        anom: kep.anom,
    })
}

#[pyfunction]
fn get_timestring_to_jd(timestring: String) -> PyResult<f64> {
    Ok(timestring_to_jd(&timestring))
}

#[pymodule]
fn cara_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_2d_foster, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pc_circle, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_eci_to_ric, m)?)?;
    m.add_function(wrap_pyfunction!(cart_to_keplerian, m)?)?;
    m.add_function(wrap_pyfunction!(get_timestring_to_jd, m)?)?;
    m.add_class::<KeplerianElements>()?;
    Ok(())
}
