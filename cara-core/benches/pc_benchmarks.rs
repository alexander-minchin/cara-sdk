use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cara_core::probability_of_collision::sdmc::pc_sdmc;
use cara_core::probability_of_collision::circle::{pc_circle, PcCircleEstimationMode};
use cara_core::probability_of_collision::foster::{pc_2d_foster, HbrType};
use nalgebra::{Vector3, Matrix6, Matrix3};

fn bench_pc_methods(c: &mut Criterion) {
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let v1 = Vector3::new(0.0, 7.5, 0.0);
    let mut c1 = Matrix6::zeros();
    c1[(0,0)] = 0.01; c1[(1,1)] = 0.01; c1[(2,2)] = 0.01;
    
    let r2 = Vector3::new(7000.0, 0.001, 0.0);
    let v2 = Vector3::new(0.0, 0.0, 7.5);
    let mut c2 = Matrix6::zeros();
    c2[(0,0)] = 0.01; c2[(1,1)] = 0.01; c2[(2,2)] = 0.01;
    
    let cov1_3x3 = Matrix3::new(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01);
    let cov2_3x3 = Matrix3::new(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01);
    
    // ndarray for foster
    use ndarray::Array2;
    let cov1_ndarray = Array2::from_elem((3, 3), 0.01);
    let cov2_ndarray = Array2::from_elem((3, 3), 0.01);

    let hbr = 0.01;

    let mut group = c.benchmark_group("Pc Algorithms");

    group.bench_function("Foster (2D Analytical)", |b| {
        b.iter(|| pc_2d_foster(
            black_box(&r1), black_box(&v1), black_box(&cov1_ndarray),
            black_box(&r2), black_box(&v2), black_box(&cov2_ndarray),
            black_box(hbr), black_box(1e-8), black_box(HbrType::Circle)
        ))
    });

    group.bench_function("SDMC (100k trials)", |b| {
        b.iter(|| pc_sdmc(
            black_box(&r1), black_box(&v1), black_box(&c1),
            black_box(&r2), black_box(&v2), black_box(&c2),
            black_box(hbr), black_box(100_000), black_box(42)
        ))
    });

    group.bench_function("Circle (64 nodes)", |b| {
        b.iter(|| pc_circle(
            black_box(&r1), black_box(&v1), black_box(&cov1_3x3),
            black_box(&r2), black_box(&v2), black_box(&cov2_3x3),
            black_box(hbr), black_box(PcCircleEstimationMode::GaussChebyshev(64))
        ))
    });

    group.finish();
}

criterion_group!(benches, bench_pc_methods);
criterion_main!(benches);
