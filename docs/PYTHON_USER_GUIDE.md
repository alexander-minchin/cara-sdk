# CARA Python User Guide

This guide provides detailed examples of how to use the CARA Rust SDK from Python, along with an explanation of the various algorithms, their trade-offs, and use cases.

## Getting Started

First, ensure you have built the library using `maturin develop` as described in the root `README.md`.

```python
import numpy as np
import cara_py
```

## Probability of Collision (Pc) Methods

The SDK provides multiple methods for calculating the Probability of Collision between two objects. Choosing the right method depends on your requirements for speed, precision, and handling of non-linear dynamics.

### 1. 2D Foster Method
The most common analytical approach for conjunction assessment.

*   **How it works:** Projects the 3D relative position and combined covariance onto a 2D "conjunction plane" (perpendicular to the relative velocity vector). It assumes linear relative motion and constant covariance during the brief encounter.
*   **Use Case:** Standard operational conjunction assessment for Earth-orbiting satellites.
*   **Trade-offs:** 
    *   **Pros:** Extremely fast and numerically stable.
    *   **Cons:** Less accurate for very large objects or extremely slow relative velocities where the linear motion assumption breaks down.

```python
pc = cara_py.compute_2d_foster(r1, v1, cov1, r2, v2, cov2, hbr)
print(f"2D Pc (Foster): {pc:.2e}")
```

### 2. Pc Circle (Gauss-Chebyshev)
A high-precision numerical integration method.

*   **How it works:** Integrates the 2D Gaussian probability density function over a circular Hard Body Radius (HBR) on the conjunction plane using Gauss-Chebyshev quadrature.
*   **Use Case:** When high numerical precision is required for circular HBRs, or for validating analytical approximations.
*   **Trade-offs:**
    *   **Pros:** More precise than simple analytical approximations for circular regions.
    *   **Cons:** Slightly slower than Foster due to numerical quadrature (though still very fast in Rust).

```python
pc_circle = cara_py.compute_pc_circle(r1, v1, cov1, r2, v2, cov2, hbr)
print(f"2D Pc (Circle): {pc_circle:.2e}")
```

### 3. Pc Elrod (Error Function Method)
An efficient analytical method using Chebyshev Gaussian Quadrature.

*   **How it works:** Uses a specialized transformation and the error function (`erf`) to reduce the 2D integral to a 1D integral, which is then solved via Chebyshev quadrature.
*   **Use Case:** High-speed calculation where analytical precision is preferred over raw numerical integration.
*   **Trade-offs:**
    *   **Pros:** Very accurate and typically faster than `PcCircle`.
    *   **Cons:** Inherits the same 2D assumptions (linear motion, constant covariance) as Foster.

```python
pc_elrod = cara_py.compute_pc_elrod(r1, v1, cov1, r2, v2, cov2, hbr)
print(f"2D Pc (Elrod): {pc_elrod:.2e}")
```

### 4. Scaled Distance Monte Carlo (SDMC)
A high-fidelity stochastic simulation method.

*   **How it works:** Randomly samples thousands (or millions) of states from the primary and secondary covariances. It propagates each sampled pair using rectilinear dynamics and checks for the minimum distance. The Pc is the ratio of "hits" (distance < HBR) to total trials.
*   **Use Case:** Validating analytical results ("Truth" source), handling cases where analytical assumptions might fail, and providing confidence intervals.
*   **Trade-offs:**
    *   **Pros:** No 2D projection assumptions; captures the full 3D nature of the encounter.
    *   **Cons:** Computationally expensive. Requires a large number of trials (e.g., 10^5 to 10^7) to resolve very small Pc values (e.g., 10^-6).

```python
# Number of trials and a random seed for reproducibility
num_trials = 100_000
seed = 42
pc_sdmc = cara_py.compute_pc_sdmc(r1, v1, cov1_6x6, r2, v2, cov2_6x6, hbr, num_trials, seed)
print(f"Pc (SDMC): {pc_sdmc:.2e}")
```

## Conjunction Data Messages (CDM)

The SDK can parse standard CCSDS Conjunction Data Messages in KVN format.

```python
cdm_path = "path/to/your_file.cdm"

# 1. Parse into a Python object for inspection
cdm = cara_py.parse_cdm(cdm_path)
print(f"TCA: {cdm.header.tca}")
print(f"Miss distance: {cdm.header.miss_distance} m")

# 2. Directly compute Pc from the file
pc = cara_py.compute_pc_from_cdm(cdm_path)
print(f"Calculated Pc: {pc:.2e}")
```

## Coordinate Transformations

### ECI to RIC Rotation
Rotate a covariance matrix from Earth-Centered Inertial (ECI) to Radial-Intrack-Cross-track (RIC). RIC is the standard frame for analyzing satellite conjunctions as it aligns with the orbit geometry.

```python
ric_cov = cara_py.rotate_eci_to_ric(cov1, r1, v1)
```

### Cartesian to Keplerian
Convert state vectors (pos/vel) to classical orbital elements.

```python
kep = cara_py.cart_to_keplerian(cart_vector)
print(f"Semi-major axis: {kep.a:.2f} km, Eccentricity: {kep.e:.4f}")
```

## Advanced Utilities

### Julian Date Conversion
The SDK provides high-performance conversion from ISO timestrings to Julian Dates, which are used internally for orbital mechanics.

```python
jd = cara_py.get_timestring_to_jd("2026-04-06 12:00:00")
```

### Covariance Remediation
Non-Positive Definite (NPD) covariances can occur due to measurement errors or numerical issues. All Pc methods in this SDK automatically apply **Eigenvalue Clipping** remediation:
1.  Perform eigen-decomposition.
2.  Clip negative or extremely small eigenvalues to a safe minimum threshold.
3.  Reconstruct the covariance matrix.
This ensures that the algorithms do not fail or produce "NaN" results due to invalid input math.
