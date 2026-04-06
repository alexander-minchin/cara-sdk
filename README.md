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
  - 2D Foster Method.
  - Pc Circle (Gauss-Chebyshev Quadrature).
  - Covariance Remediation (Eigenvalue Clipping).
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

## License

MIT OR Apache-2.0
