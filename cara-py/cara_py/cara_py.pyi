import numpy as np
from typing import List, Optional, Protocol

class KeplerianElements:
    a: float
    e: float
    i: float
    omega: float
    w: float
    anom: float

class PyCdmHeader:
    version: str
    tca: str
    miss_distance: float
    hbr: Optional[float]
    collision_probability: float

class PyCdmObject:
    object_name: str
    x: float
    y: float
    z: float
    x_dot: float
    y_dot: float
    z_dot: float

class PyCdm:
    header: PyCdmHeader
    object1: PyCdmObject
    object2: PyCdmObject

class PyEdfPValues:
    w2_p: float
    u2_p: float
    a2_p: float

class PyPcDilutionOutput:
    pc_one: float
    diluted: bool
    pc_max: float
    sf_max: float
    pc_buffer: List[float]
    sf_buffer: List[float]
    converged: bool
    iterations: int

class PyCollisionConsequence:
    is_catastrophic: bool
    num_pieces: int

def compute_2d_foster(
    r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
    r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
    hbr: float
) -> float: ...

def compute_pc_circle(
    r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
    r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
    hbr: float
) -> float: ...

def compute_pc_elrod(
    r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
    r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
    hbr: float
) -> float: ...

def compute_pc_sdmc(
    r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
    r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
    hbr: float, num_trials: int, seed: int
) -> float: ...

def compute_frisbee_max_pc(
    r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
    r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
    hbr: float
) -> float: ...

def compute_pc_dilution(
    r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
    r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
    hbr: float, scaling: str
) -> PyPcDilutionOutput: ...

def compute_pc_from_cdm(path: str) -> float: ...
def compute_pc_circle_from_cdm(path: str) -> float: ...
def compute_pc_elrod_from_cdm(path: str) -> float: ...

def rotate_eci_to_ric(eci_cov: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray: ...

def cart_to_keplerian(cart: np.ndarray) -> KeplerianElements: ...

def get_timestring_to_jd(timestring: str) -> float: ...

def parse_cdm(path: str) -> PyCdm: ...

def test_normality(data: np.ndarray, mean: float, std_dev: float) -> PyEdfPValues: ...

def compute_collision_consequence(
    primary_mass: float, v_rel: float, secondary_mass: float, lc: Optional[float]
) -> PyCollisionConsequence: ...

def get_nasa_sem_size(rcs_normalized: float) -> float: ...
