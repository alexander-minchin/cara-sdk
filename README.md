# CARA SDK (Rust Port)

A high-performance Rust port of the NASA CARA (Conjunction Assessment Risk Analysis) Analysis Tools. This project provides a memory-safe, thread-safe, and highly optimized implementation of standard astrodynamics algorithms and probability of collision (Pc) methods.

## Project Structure

- `cara-core/`: Pure Rust implementation of the core algorithms.
- `cara-py/`: Python bindings using PyO3 and Maturin.
- `cara-cabi/`: C ABI export layer (planned).
- `cara-java/`: Java bindings via JNI (planned).
- `cara-wasm/`: WebAssembly bindings (planned).
- `upstream/`: Git submodule pointing to the original MATLAB implementation.

## Features

- **Probability of Collision (Pc):**
  - **2D Foster Method:** Standard analytical approach.
  - **Pc Circle:** High-precision numerical integration (Gauss-Chebyshev).
  - **Pc Elrod:** Efficient analytical error function method.
  - **Pc SDMC:** High-fidelity Scaled Distance Monte Carlo.
  - **Maximum Pc (Frisbee):** Calculates upper bound Pc when covariances are incomplete.
  - **Pc Dilution Analysis:** Iterative search for maximum Pc and dilution region detection.
  - **Covariance Remediation:** Automatic eigenvalue clipping for numerical stability.
- **Collision Consequence:**
  - **Debris Estimation:** Predicts the number of pieces generated in a collision using the NASA Breakup Model.
  - **Size Estimation:** Implementation of the NASA Size Estimation Model (SEM) for RCS-to-size conversion.
- **Covariance Realism:**
  - **Normality Tests:** Cramer-von Mises and Anderson-Darling goodness-of-fit tests.
  - **Residual Evaluation:** Analyze if observed residuals conform to predicted covariances.
- **CDM Analysis:**
  - **KVN Parser:** Read standard CCSDS Conjunction Data Messages.
  - **Automated Calculation:** Directly compute Pc from CDM files.
- **Orbit Transformations:**
  - Cartesian to/from Keplerian.
  - Cartesian to Equinoctial elements.
  - Coordinate rotations (ECI to RIC).
- **Utilities:**
  - Julian Date conversions.
  - Vectorized error function differences (`erf_dif`).
  - Gauss-Chebyshev node generation.

## Installation

### Prerequisites

- Rust Toolchain (stable)
- Python 3.7+
- `pip` and `venv`

### Building the Python Library

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install build dependencies
pip install maturin numpy

# Build and install the package
cd cara-py
maturin develop
```

## Documentation

- [Python User Guide](docs/PYTHON_USER_GUIDE.md) - Detailed usage and examples.

## Licensing and Attribution

This project is a derivative work of the **NASA CARA Analysis Tools**, originally developed by the Conjunction Assessment Risk Analysis (CARA) team at the National Aeronautics and Space Administration (NASA).

### Original Source
- **Original Project:** [NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools)
- **Original License:** NASA Open Source Agreement (NOSA)

### This Port
The Rust implementation and associated bindings in this repository are provided under the **NASA Open Source Agreement (NOSA) version 1.3**. 

By using, distributing, or contributing to this project, you agree to abide by the terms of the NOSA. Copies of the applicable NASA Open Source Software Agreements can be found in the `legal/` directory:
- `legal/NOSA_GSC-18593-1.pdf`
- `legal/NOSA_GSC-18848-1.pdf`
- `legal/NOSA_GSC-19374-1.pdf`

## Disclaimer
This software is provided "as is" without any warranty of any kind, either expressed, implied, or statutory.
