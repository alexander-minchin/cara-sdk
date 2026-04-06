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

### 5. Maximum 2D Pc (Frisbee Method)
Calculates the maximum possible Pc value given the current geometry, effectively assuming the "worst-case" covariance.

*   **How it works:** Based on Frisbee (2012), this method identifies the covariance scaling that would maximize the collision probability for the given miss distance. It handles cases where one or both covariances might be poorly known.
*   **Use Case:** Identifying the theoretical ceiling of risk for a conjunction.
*   **Trade-offs:** 
    *   **Pros:** Provides a definitive upper bound.
    *   **Cons:** Often highly conservative (overestimates risk).

```python
max_pc = cara_py.compute_frisbee_max_pc(r1, v1, cov1, r2, v2, cov2, hbr)
print(f"Max Pc (Frisbee): {max_pc:.2e}")
```

### 6. Pc Dilution Analysis
Analyzes how the Pc changes as the covariance is scaled, identifying if the event is in the "dilution region."

*   **How it works:** Iteratively scales the input covariances by a factor `Sf` and calculates the resulting Pc. If scaling the covariance *down* increases the Pc, the event is "diluted" (meaning the current uncertainty is so large it's actually hiding the risk).
*   **Use Case:** Determining if a low Pc is "robust" (low risk because objects are far apart) or "diluted" (low risk only because uncertainty is high).
*   **Trade-offs:**
    *   **Pros:** Essential for operational decision-making to avoid ignoring high-risk diluted events.
    *   **Cons:** Requires multiple Pc calculations (iterative).

```python
# scaling can be "both", "primary", or "secondary"
dilution = cara_py.compute_pc_dilution(r1, v1, cov1, r2, v2, cov2, hbr, "both")

print(f"Nominal Pc: {dilution.pc_one:.2e}")
print(f"Maximum Pc: {dilution.pc_max:.2e}")
print(f"Scale Factor at Max: {dilution.sf_max:.4f}")
print(f"Is Diluted: {dilution.diluted}")
```

## Collision Consequence

Estimates the physical results of a potential collision between two objects.

### 1. Debris Piece Estimation (NASA Breakup Model)
Predicts the number of debris pieces larger than a characteristic length `Lc` that would be generated.

*   **How it works:** Uses the standard NASA ODPO (Orbital Debris Program Office) breakup model. It calculates collision energy to determine if the impact is "catastrophic" (total breakup) or "non-catastrophic" and then scales the resulting debris count based on mass and velocity.
*   **Use Case:** Assessing the environmental impact and long-term risk to other satellites if a collision occurs.

```python
# Masses in kg, Relative Velocity in m/s
primary_mass = 1500.0
secondary_mass = 5.0
v_rel = 14500.0 # 14.5 km/s
lc = 0.05 # 5cm characteristic length

consequence = cara_py.compute_collision_consequence(primary_mass, v_rel, secondary_mass, lc)

print(f"Is Catastrophic: {consequence.is_catastrophic}")
print(f"Expected Debris Pieces (>5cm): {consequence.num_pieces}")
```

### 2. Size Estimation (NASA SEM)
Converts Radar Cross Section (RCS) values into physical characteristic sizes.

*   **How it works:** Implements the NASA Size Estimation Model (SEM), which uses piecewise fits across the Rayleigh, Mie, and Optical regimes to relate normalized RCS to physical diameter.
*   **Use Case:** Estimating object dimensions and mass when only radar data is available.

```python
# Normalized RCS (RCS / lambda^2)
rcs_norm = 0.012
size_norm = cara_py.get_nasa_sem_size(rcs_norm)
print(f"Normalized Size (d/lambda): {size_norm:.4f}")
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

## Covariance Realism

Covariance realism analysis assesses whether predicted covariances accurately represent the actual distribution of observed residuals.

### Normality Tests (Goodness-of-Fit)

The SDK provides high-performance implementations of Empirical Distribution Function (EDF) tests to check if data follows a standard Normal distribution.

*   **Cramer-von Mises (W2):** Measures the overall fit of the distribution.
*   **Anderson-Darling (A2):** Places more weight on the "tails" of the distribution, making it sensitive to outliers.

```python
# Sample data (e.g., normalized residuals)
data = np.random.normal(0, 1, 100)

# Perform normality test against N(0, 1)
results = cara_py.test_normality(data, mean=0.0, std_dev=1.0)

print(f"Cramer-von Mises p-value: {results.w2_p:.4f}")
print(f"Anderson-Darling p-value: {results.a2_p:.4f}")

if results.a2_p < 0.05:
    print("Warning: Data significantly deviates from a normal distribution.")
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
