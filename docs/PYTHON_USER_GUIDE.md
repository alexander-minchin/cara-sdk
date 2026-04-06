# CARA Python User Guide

This guide provides detailed examples of how to use the CARA Rust SDK from Python.

## Getting Started

First, ensure you have built the library using `maturin develop` as described in the root `README.md`.

```python
import numpy as np
import cara_py
```

## Probability of Collision (Pc)

The SDK provides multiple methods for calculating the Probability of Collision between two objects.

### 2D Foster Method

The Foster method is a standard 2D Pc calculation. It projects the combined covariance onto the conjunction plane.

```python
# Primary object state (km, km/s)
r1 = np.array([7000.0, 0.0, 0.0])
v1 = np.array([0.0, 7.5, 0.0])
cov1 = np.diag([0.01, 0.01, 0.01]) # 3x3 position covariance

# Secondary object state
r2 = np.array([7000.0, 0.001, 0.0])
v2 = np.array([0.0, 0.0, 7.5])
cov2 = np.diag([0.01, 0.01, 0.01])

hbr = 0.01 # Hard Body Radius in km (10 meters)

pc = cara_py.compute_2d_foster(r1, v1, cov1, r2, v2, cov2, hbr)
print(f"2D Pc (Foster): {pc:.2e}")
```

### Pc Circle (Gauss-Chebyshev)

The Circle method uses Gauss-Chebyshev quadrature for high-precision integration over a circular region on the conjunction plane.

```python
pc_circle = cara_py.compute_pc_circle(r1, v1, cov1, r2, v2, cov2, hbr)
print(f"2D Pc (Circle): {pc_circle:.2e}")
```

## Coordinate Transformations

### ECI to RIC Rotation

Rotate a covariance matrix from the Earth-Centered Inertial (ECI) frame to the Radial-Intrack-Cross-track (RIC) frame.

```python
# Rotation based on position and velocity
ric_cov = cara_py.rotate_eci_to_ric(cov1, r1, v1)

print("RIC Covariance Matrix:")
print(ric_cov)
```

### Cartesian to Keplerian

Convert Cartesian state vectors to Keplerian orbital elements.

```python
# Combined state vector [x, y, z, vx, vy, vz]
cart = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])

kep = cara_py.cart_to_keplerian(cart)

print(f"Semi-major axis (a): {kep.a:.2f} km")
print(f"Eccentricity (e): {kep.e:.4f}")
print(f"Inclination (i): {kep.i:.2f} deg")
print(f"RAAN (Omega): {kep.omega:.2f} deg")
print(f"Arg of Perigee (w): {kep.w:.2f} deg")
print(f"Mean Anomaly (M): {kep.anom:.2f} deg")
```

## Time Utilities

### Julian Date Conversion

Convert standard time strings to Julian Dates (JD).

```python
timestring = "2026-04-06 12:00:00"
jd = cara_py.get_timestring_to_jd(timestring)

print(f"Julian Date: {jd}")
```

The function supports both `YYYY-MM-DD HH:MM:SS` and `YYYY-MM-DDTHH:MM:SS` formats.

## Advanced Usage

### Handling Large Conjunction Sets

While the current Python API focuses on single conjunctions, the underlying Rust core is designed for high-performance batch processing. If you need to process thousands of conjunctions, consider using multi-threading in Python or waiting for the upcoming vectorized Python API.

### Covariance Remediation

The Pc functions automatically perform eigenvalue clipping remediation if the input covariance is not positive definite. This ensures numerical stability in edge cases.
