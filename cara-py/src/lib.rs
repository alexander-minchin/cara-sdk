use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use ndarray::Array2;
use cara_core::probability_of_collision::foster::{pc_2d_foster, HbrType};
use cara_core::probability_of_collision::circle::{pc_circle, PcCircleEstimationMode};
use cara_core::probability_of_collision::elrod::pc_elrod;
use cara_core::probability_of_collision::sdmc::pc_sdmc;
use cara_core::utils::time_transformations::timestring_to_jd;
use cara_core::utils::orbit_transformations::{cart_to_kep, eci_to_ric, ric_to_eci, AnomalyType, UnitType};
use cara_core::cdm::{parse_cdm_kvn};
use nalgebra::{Vector3, Vector6, Matrix3, Matrix6};

#[pyclass]
#[derive(Clone)]
struct PyCdmHeader {
    #[pyo3(get)]
    pub version: String,
    #[pyo3(get)]
    pub tca: String,
    #[pyo3(get)]
    pub miss_distance: f64,
    #[pyo3(get)]
    pub hbr: Option<f64>,
    #[pyo3(get)]
    pub collision_probability: f64,
}

#[pyclass]
#[derive(Clone)]
struct PyCdmObject {
    #[pyo3(get)]
    pub object_name: String,
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
    #[pyo3(get)]
    pub z: f64,
    #[pyo3(get)]
    pub x_dot: f64,
    #[pyo3(get)]
    pub y_dot: f64,
    #[pyo3(get)]
    pub z_dot: f64,
}

#[pyclass]
#[derive(Clone)]
struct PyCdm {
    #[pyo3(get)]
    pub header: PyCdmHeader,
    #[pyo3(get)]
    pub object1: PyCdmObject,
    #[pyo3(get)]
    pub object2: PyCdmObject,
}

#[pyfunction]
fn parse_cdm(path: String) -> PyResult<PyCdm> {
    let cdm = parse_cdm_kvn(&path).map_err(|e| PyRuntimeError::new_err(e))?;
    
    Ok(PyCdm {
        header: PyCdmHeader {
            version: cdm.header.version,
            tca: cdm.header.tca,
            miss_distance: cdm.header.miss_distance,
            hbr: cdm.header.hbr,
            collision_probability: cdm.header.collision_probability,
        },
        object1: PyCdmObject {
            object_name: cdm.object1.object_name,
            x: cdm.object1.x,
            y: cdm.object1.y,
            z: cdm.object1.z,
            x_dot: cdm.object1.x_dot,
            y_dot: cdm.object1.y_dot,
            z_dot: cdm.object1.z_dot,
        },
        object2: PyCdmObject {
            object_name: cdm.object2.object_name,
            x: cdm.object2.x,
            y: cdm.object2.y,
            z: cdm.object2.z,
            x_dot: cdm.object2.x_dot,
            y_dot: cdm.object2.y_dot,
            z_dot: cdm.object2.z_dot,
        },
    })
}

#[pyfunction]
fn compute_pc_from_cdm(path: String) -> PyResult<f64> {
    let cdm = parse_cdm_kvn(&path).map_err(|e| PyRuntimeError::new_err(e))?;
    
    let r1 = Vector3::new(cdm.object1.x, cdm.object1.y, cdm.object1.z);
    let v1 = Vector3::new(cdm.object1.x_dot, cdm.object1.y_dot, cdm.object1.z_dot);
    let r2 = Vector3::new(cdm.object2.x, cdm.object2.y, cdm.object2.z);
    let v2 = Vector3::new(cdm.object2.x_dot, cdm.object2.y_dot, cdm.object2.z_dot);
    
    // CDM covariances are in RIC frame (m^2). Convert to ECI (or parent frame) and km^2.
    let cov1_ndarray = (Array2::from_shape_vec((6, 6), cdm.object1.covariance.iter().cloned().collect()).unwrap()) / 1e6;
    let cov2_ndarray = (Array2::from_shape_vec((6, 6), cdm.object2.covariance.iter().cloned().collect()).unwrap()) / 1e6;
    
    let cov1_eci_full = ric_to_eci(&cov1_ndarray, &r1, &v1, true);
    let cov2_eci_full = ric_to_eci(&cov2_ndarray, &r2, &v2, true);

    // Extract 3x3 position part in ECI
    let cov1_eci_3x3 = cov1_eci_full.slice(ndarray::s![0..3, 0..3]).to_owned();
    let cov2_eci_3x3 = cov2_eci_full.slice(ndarray::s![0..3, 0..3]).to_owned();

    let hbr_m = cdm.header.hbr.unwrap_or(15.0); 
    let hbr_km = hbr_m / 1000.0;

    let output = pc_2d_foster(
        &r1, &v1, &cov1_eci_3x3,
        &r2, &v2, &cov2_eci_3x3,
        hbr_km, 1e-8, HbrType::Circle
    );

    Ok(output.pc)
}

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
fn compute_pc_elrod(
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

    let output = pc_elrod(
        &r1_vec, &v1_vec, &cov1_mat,
        &r2_vec, &v2_vec, &cov2_mat,
        hbr, 64
    );

    Ok(output.pc)
}

#[pyfunction]
fn compute_pc_sdmc(
    r1: PyReadonlyArray1<f64>,
    v1: PyReadonlyArray1<f64>,
    cov1: PyReadonlyArray2<f64>,
    r2: PyReadonlyArray1<f64>,
    v2: PyReadonlyArray1<f64>,
    cov2: PyReadonlyArray2<f64>,
    hbr: f64,
    num_trials: usize,
    seed: u64,
) -> PyResult<f64> {
    let r1_vec = Vector3::from_column_slice(r1.as_slice()?);
    let v1_vec = Vector3::from_column_slice(v1.as_slice()?);
    let r2_vec = Vector3::from_column_slice(r2.as_slice()?);
    let v2_vec = Vector3::from_column_slice(v2.as_slice()?);

    let cov1_mat = Matrix6::from_iterator(cov1.to_owned_array().iter().cloned());
    let cov2_mat = Matrix6::from_iterator(cov2.to_owned_array().iter().cloned());

    let output = pc_sdmc(
        &r1_vec, &v1_vec, &cov1_mat,
        &r2_vec, &v2_vec, &cov2_mat,
        hbr, num_trials, seed
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

#[pyfunction]
fn compute_pc_circle_from_cdm(path: String) -> PyResult<f64> {
    let cdm = parse_cdm_kvn(&path).map_err(|e| PyRuntimeError::new_err(e))?;
    let r1 = Vector3::new(cdm.object1.x, cdm.object1.y, cdm.object1.z);
    let v1 = Vector3::new(cdm.object1.x_dot, cdm.object1.y_dot, cdm.object1.z_dot);
    let r2 = Vector3::new(cdm.object2.x, cdm.object2.y, cdm.object2.z);
    let v2 = Vector3::new(cdm.object2.x_dot, cdm.object2.y_dot, cdm.object2.z_dot);
    let cov1_ndarray = (Array2::from_shape_vec((6, 6), cdm.object1.covariance.iter().cloned().collect()).unwrap()) / 1e6;
    let cov2_ndarray = (Array2::from_shape_vec((6, 6), cdm.object2.covariance.iter().cloned().collect()).unwrap()) / 1e6;
    let cov1_eci_full = ric_to_eci(&cov1_ndarray, &r1, &v1, true);
    let cov2_eci_full = ric_to_eci(&cov2_ndarray, &r2, &v2, true);
    let cov1_eci_3x3 = Matrix3::from_iterator(cov1_eci_full.slice(ndarray::s![0..3, 0..3]).iter().cloned());
    let cov2_eci_3x3 = Matrix3::from_iterator(cov2_eci_full.slice(ndarray::s![0..3, 0..3]).iter().cloned());
    let hbr_m = cdm.header.hbr.unwrap_or(15.0); 
    let hbr_km = hbr_m / 1000.0;
    let output = pc_circle(&r1, &v1, &cov1_eci_3x3, &r2, &v2, &cov2_eci_3x3, hbr_km, PcCircleEstimationMode::GaussChebyshev(64));
    Ok(output.pc)
}

#[pyfunction]
fn compute_pc_elrod_from_cdm(path: String) -> PyResult<f64> {
    let cdm = parse_cdm_kvn(&path).map_err(|e| PyRuntimeError::new_err(e))?;
    let r1 = Vector3::new(cdm.object1.x, cdm.object1.y, cdm.object1.z);
    let v1 = Vector3::new(cdm.object1.x_dot, cdm.object1.y_dot, cdm.object1.z_dot);
    let r2 = Vector3::new(cdm.object2.x, cdm.object2.y, cdm.object2.z);
    let v2 = Vector3::new(cdm.object2.x_dot, cdm.object2.y_dot, cdm.object2.z_dot);
    let cov1_ndarray = (Array2::from_shape_vec((6, 6), cdm.object1.covariance.iter().cloned().collect()).unwrap()) / 1e6;
    let cov2_ndarray = (Array2::from_shape_vec((6, 6), cdm.object2.covariance.iter().cloned().collect()).unwrap()) / 1e6;
    let cov1_eci_full = ric_to_eci(&cov1_ndarray, &r1, &v1, true);
    let cov2_eci_full = ric_to_eci(&cov2_ndarray, &r2, &v2, true);
    let cov1_eci_3x3 = Matrix3::from_iterator(cov1_eci_full.slice(ndarray::s![0..3, 0..3]).iter().cloned());
    let cov2_eci_3x3 = Matrix3::from_iterator(cov2_eci_full.slice(ndarray::s![0..3, 0..3]).iter().cloned());
    let hbr_m = cdm.header.hbr.unwrap_or(15.0); 
    let hbr_km = hbr_m / 1000.0;
    let output = pc_elrod(&r1, &v1, &cov1_eci_3x3, &r2, &v2, &cov2_eci_3x3, hbr_km, 64);
    Ok(output.pc)
}

#[pymodule]
fn cara_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_2d_foster, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pc_circle, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pc_elrod, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pc_sdmc, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pc_from_cdm, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pc_circle_from_cdm, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pc_elrod_from_cdm, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_eci_to_ric, m)?)?;
    m.add_function(wrap_pyfunction!(cart_to_keplerian, m)?)?;
    m.add_function(wrap_pyfunction!(get_timestring_to_jd, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cdm, m)?)?;
    m.add_class::<KeplerianElements>()?;
    m.add_class::<PyCdm>()?;
    m.add_class::<PyCdmHeader>()?;
    m.add_class::<PyCdmObject>()?;
    Ok(())
}
