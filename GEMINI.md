---
name: CARA Rust Port Plan
overview: Architecture and project structure for porting CARA_Analysis_Tools from MATLAB to Rust, with multi-language bindings (Python, Java, WASM), upstream tracking via git submodule, and a phased approach toward SIMD optimization. Built specifically to be easily pip-installed and executed from a Python script.
todos:
  - id: repo-scaffold
    content: Create new cara-sdk Cargo workspace repo with git submodule for CARA_Analysis_Tools and directory skeleton
    status: pending
  - id: cargo-setup
    content: Set up workspace Cargo.toml with members for core, cabi, and language bindings
    status: pending
  - id: port-utils
    content: Port DistributedMatlab/Utils/ (104 .m files) to Rust cara-core/src/utils/
    status: pending
  - id: port-utils-tests
    content: Port Utils unit tests to native cargo test in cara-core/tests/ using the approx crate
    status: pending
  - id: port-pc
    content: Port DistributedMatlab/ProbabilityOfCollision/ (60 .m files) to Rust
    status: pending
  - id: port-pc-tests
    content: Port ProbabilityOfCollision unit tests and validate against MATLAB expected outputs
    status: pending
  - id: port-remaining
    content: Port remaining 7 modules (CovarianceRealism, ODQuality, Max2DPc, MonteCarlo, CollisionConsequence, EventRate, LightPollution)
    status: pending
  - id: cabi-layer
    content: Implement cara-cabi cdylib export layer over the Rust core API
    status: pending
  - id: python-bindings
    content: Implement Python bindings via PyO3 and Maturin in cara-py/ to make the tool directly runnable from Python
    status: pending
  - id: java-bindings
    content: Implement Java bindings via the jni crate in cara-java/
    status: pending
  - id: wasm-bindings
    content: Implement WASM build via wasm-pack and wasm-bindgen in cara-wasm/
    status: pending
  - id: simd-optimization
    content: Profile and add SIMD optimization using portable-simd (std::simd) for hot paths
    status: pending
isProject: false
---

CARA Rust Port: Project Structure, Version Control, and Tooling
0. Why Rust: Motivation and Language Choice
The motivation behind choosing Rust as the re-implementation language is to create a single, memory-safe compiled core that can be exported to multiple target languages (Python, Java, WASM) via bindings and wrappers. By porting the MATLAB code to Rust, we can leverage Cargo's robust package management and produce high-performance, safely concurrent SIMD-optimized code once, letting each binding inherit that performance without duplicating effort.
Crucially, Rust provides best-in-class tooling (PyO3 + Maturin) to compile native extensions that can be transparently imported and run from a standard Python script, fulfilling the primary deployment requirement.
Why Rust works well for this cross-platform, SIMD-export vision
 * Safety and Concurrency: Rust's strict borrow checker eliminates data races and memory leaks at compile time. This makes multi-threading Monte Carlo simulations and heavy matrix computations inherently safer than C/C++.
 * Python Integration (PyO3 + Maturin): This combination is the gold standard for scientific Python extensions. It compiles the Rust core directly into a wheel (.whl) that can be pip installed and run from a Python script seamlessly, translating Rust panics into Python exceptions and bridging numpy arrays to Rust types.
 * WASM: wasm-pack and wasm-bindgen provide an incredibly streamlined pipeline for compiling numerical Rust code to WebAssembly, making browser-based execution highly performant.
 * Scientific computing ecosystem: Crates like ndarray map cleanly to MATLAB/NumPy N-dimensional array semantics (broadcasting, slicing), while nalgebra provides astrodynamics-grade linear algebra.
 * SIMD \u2014 optimize once, export everywhere: You write SIMD-optimized Rust once using portable-simd (or heavily relying on LLVM autovectorization), and every binding inherits it.
Architectural note: the C ABI layer
Even though the internals are Rust, exposing a flat C ABI (extern "C") via a cdylib crate gives a universal FFI surface that any future language can consume without native bindings (similar to the Sgp4Prop pattern). PyO3 and JNI can bypass this and bind directly to Rust structs for richer APIs, but having the C ABI as a fallback is cheap insurance.
1. Version Control: Git Submodule Strategy
Use git submodule to embed CARA_Analysis_Tools inside the new repo. This directly supports pinning to specific upstream commits and tracking MATLAB changes:
git submodule add https://github.com/nasa/CARA_Analysis_Tools.git upstream
git submodule update --init

To operationalize the diff-and-reconcile step, maintain a BASELINE_COMMIT file in your repo. A script can then automate diffing your pinned version against the latest upstream master:
cd upstream
git diff $(cat ../BASELINE_COMMIT)..HEAD -- DistributedMatlab/

2. Recommended Project Structure (Cargo Workspace)
cara-sdk/
├── Cargo.toml                   # Root Cargo workspace configuration
├── BASELINE_COMMIT              # SHA of upstream commit used for current Rust impl
├── upstream/                    # git submodule -> CARA_Analysis_Tools
│
├── cara-core/                   # Pure Rust implementation
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # Public Rust API
│   │   ├── probability_of_collision/
│   │   ├── covariance_realism/
│   │   ├── event_rate/
│   │   ├── utils/
│   │   └── cdm.rs
│   └── tests/                   # Ported MATLAB unit tests (native cargo test)
│       ├── test_pc_2d_foster.rs
│       ├── test_covariance_transforms.rs
│       └── fixtures/            # CDM files, test inputs from upstream/DataFiles
│
├── cara-cabi/                   # C ABI export layer (cdylib)
│   ├── Cargo.toml               # depends on cara-core
│   ├── build.rs                 # generates C headers using cbindgen
│   └── src/lib.rs               # Flat C function signatures (extern "C")
│
├── cara-py/                     # Python bindings (PyO3 + Maturin)
│   ├── Cargo.toml               # crate-type = ["cdylib"]
│   ├── pyproject.toml           # Maturin build config
│   ├── src/lib.rs               # PyO3 wrappers exposing cara-core
│   └── cara/                    # Python package wrapper (optional pure Python stubs)
│       ├── __init__.py
│       └── run_cara.py          # Entry point for running the tool from Python
│
├── cara-java/                   # Java bindings (jni crate)
│   ├── Cargo.toml
│   ├── src/lib.rs               # JNI exports
│   └── java/                    # Java source and build files (Maven/Gradle)
│
├── cara-wasm/                   # WASM bindings (wasm-bindgen)
│   ├── Cargo.toml
│   ├── src/lib.rs               # #[wasm_bindgen] exports
│   └── package.json
│
└── scripts/
    ├── diff_upstream.sh         # Automate baseline diff
    ├── port_status.py           # Track which MATLAB functions are ported
    └── generate_test_fixtures.m # Extract test data from MATLAB

3. Rust Tooling
Compiler and Build
 * Cargo as the primary build system and workspace manager.
 * Maturin to orchestrate the Python builds (compiling the Rust code directly into Python wheels).
 * cbindgen in the cara-cabi build script to automatically generate cara_c.h for external C consumers.
Testing
 * cargo test as the built-in test runner.
 * Port the ~38 MATLAB *_UnitTest.m files systematically into tests/.
 * Use the approx crate for tolerance-based floating-point assertions (essential for matching MATLAB's Pc values).
 * Extract test fixture data from upstream/DataFiles/ into cara-core/tests/fixtures/.
SIMD Roadmap
When the port is complete and tested:
 * Utilize std::simd (portable-simd) on nightly Rust, or heavily optimize for LLVM autovectorization using iterator pipelines.
 * Target inner loops in Pc computation (covariance projections, numerical integration).
 * Compile target features (RUSTFLAGS="-C target-cpu=native") to ensure AVX-512/NEON instructions are utilized during compilation.
4. Binding Strategies
| Target | Mechanism | Notes |
|---|---|---|
| Python | PyO3 + Maturin | Compiles directly to a Python extension module. Exposes Rust structs to Python natively. Converts numpy arrays to Rust ndarray types seamlessly using rust-numpy. |
| Java | jni crate | Implements Java Native Interface directly in Rust. Safer than C++ JNI due to Rust's type-checking of JNI environment pointers. |
| WASM | wasm-pack + wasm-bindgen | Compiles Rust to WebAssembly. Generates all necessary JavaScript/TypeScript glue code automatically. |
5. Running from a Python Script
By using PyO3 and Maturin in the cara-py directory, the final tool becomes natively importable. Once built (maturin develop or pip install .), you can execute the compiled Rust backend via a standard Python script like this:
# scripts/run_cara.py
import numpy as np
import cara  # This is the compiled Rust binary extension

def main():
    # Example inputs (can be loaded from CDM)
    primary_state = np.array([1000.0, 2000.0, 3000.0])
    secondary_state = np.array([1050.0, 2010.0, 2990.0])
    covariance = np.eye(3) * 10.0
    
    # Calls the heavily optimized Rust core
    pc_result = cara.probability_of_collision.compute_2d_foster(
        primary_state, 
        secondary_state, 
        covariance
    )
    
    print(f"Probability of Collision: {pc_result}")

if __name__ == "__main__":
    main()

6. Phased Execution
Phase 1 \u2014 Core Rust port:
 * Set up Cargo workspace repo structure with git submodule.
 * Port MATLAB modules to Rust cara-core, starting with Utils/ (foundation), then ProbabilityOfCollision/ (most critical).
 * Port all unit tests; achieve parity with MATLAB runtests output via cargo test.
Phase 2 \u2014 Python "Runnable" Target (Priority Binding):
 * Implement cara-py via PyO3.
 * Map core Rust functions to Python/NumPy inputs.
 * Create the Python runner script to satisfy the primary execution requirement.
Phase 3 \u2014 Additional Bindings & C-ABI:
 * Implement cara-cabi C ABI layer.
 * Java bindings via jni crate.
 * WASM via wasm-pack.
Phase 4 \u2014 SIMD optimization:
 * Profile hotspots using cargo flamegraph.
 * Introduce portable-simd or optimized iterators for vectorized math.
 * Benchmark against MATLAB baseline.
7. Module Porting Priority
Based on dependency analysis of the MATLAB codebase:
 * Utils/ (104 .m files) \u2014 foundational: orbit/covariance/time/pos-vel transforms, CDM parsing.
 * ProbabilityOfCollision/ (60 .m files) \u2014 core mission: 2D/3D Pc, SDMC, Foster, Hall, Elrod.
 * CovarianceRealism/ (9 .m files) \u2014 covariance consistency checks.
 * ODQualityAssessment/ (6 .m files) \u2014 OD quality.
 * Maximum2DPc/ (5 .m files) \u2014 Frisbee dilution.
 * MonteCarloPc/ (4 .m files) \u2014 Monte Carlo (SDMC preferred).
 * CollisionConsequence/ (7 .m files) \u2014 debris/mass.
 * EventRate/ (38 .m files) \u2014 semi-empirical mission risk.
 * EvaluateLightPollution/ (~42 .m files) \u2014 constellation light pollution.
