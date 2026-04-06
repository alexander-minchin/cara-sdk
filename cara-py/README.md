# cara-py

Python bindings for the CARA Rust SDK.

## Installation

### From Source

```bash
pip install maturin numpy
cd cara-py
maturin develop
```

## Quick Example

```python
import numpy as np
import cara_py

# Cartesian to Keplerian
cart = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
kep = cara_py.cart_to_keplerian(cart)
print(f"Semi-major axis: {kep.a} km")

# 2D Pc calculation
r1 = np.array([7000.0, 0.0, 0.0])
v1 = np.array([0.0, 7.5, 0.0])
cov1 = np.diag([0.01, 0.01, 0.01])
r2 = np.array([7000.0, 0.001, 0.0])
v2 = np.array([0.0, 0.0, 7.5])
cov2 = np.diag([0.01, 0.01, 0.01])
hbr = 0.01

pc = cara_py.compute_2d_foster(r1, v1, cov1, r2, v2, cov2, hbr)
print(f"Pc: {pc}")
```

For more details, see the [Python User Guide](../docs/PYTHON_USER_GUIDE.md).
