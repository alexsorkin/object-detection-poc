use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ioutrack::hungarian::HungarianSolver;
use ndarray::Array2;
use rand::prelude::*;

fn generate_random_cost_matrix(detections: usize, tracks: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    Array2::from_shape_fn((detections, tracks), |_| rng.gen_range(0.0..1.0))
}

fn bench_hungarian_small(c: &mut Criterion) {
    let cost_matrix = generate_random_cost_matrix(10, 10);

    c.bench_function("hungarian_10x10", |b| {
        b.iter(|| HungarianSolver::solve(black_box(cost_matrix.view()), black_box(0.5)))
    });
}

fn bench_hungarian_medium(c: &mut Criterion) {
    let cost_matrix = generate_random_cost_matrix(50, 50);

    c.bench_function("hungarian_50x50", |b| {
        b.iter(|| HungarianSolver::solve(black_box(cost_matrix.view()), black_box(0.5)))
    });
}

fn bench_hungarian_large(c: &mut Criterion) {
    let cost_matrix = generate_random_cost_matrix(100, 100);

    c.bench_function("hungarian_100x100", |b| {
        b.iter(|| HungarianSolver::solve(black_box(cost_matrix.view()), black_box(0.5)))
    });
}

fn bench_hungarian_iou_conversion(c: &mut Criterion) {
    let iou_matrix = generate_random_cost_matrix(50, 50);

    c.bench_function("hungarian_iou_50x50", |b| {
        b.iter(|| HungarianSolver::solve_iou(black_box(iou_matrix.view()), black_box(0.5)))
    });
}

criterion_group!(
    benches,
    bench_hungarian_small,
    bench_hungarian_medium,
    bench_hungarian_large,
    bench_hungarian_iou_conversion
);
criterion_main!(benches);
