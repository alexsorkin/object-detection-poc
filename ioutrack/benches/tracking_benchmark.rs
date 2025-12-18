//! Benchmarks for tracking algorithms

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ioutrack::{ByteMultiTracker, SortMultiTracker};
use ndarray::Array2;
use std::hint::black_box;

fn create_test_detections(n_detections: usize, n_frames: usize) -> Vec<Array2<f32>> {
    (0..n_frames)
        .map(|frame| {
            let mut data = Vec::with_capacity(n_detections * 5);
            for i in 0..n_detections {
                let x = (frame * 10 + i * 50) as f32;
                let y = (frame * 5 + i * 30) as f32;
                data.extend(&[x, y, x + 50.0, y + 30.0, 0.8]); // [x1, y1, x2, y2, score]
            }
            Array2::from_shape_vec((n_detections, 5), data).unwrap()
        })
        .collect()
}

fn bench_sort_update(c: &mut Criterion) {
    let detections = create_test_detections(20, 10);

    c.bench_function("sort_update_20_detections", |b| {
        b.iter_batched(
            || {
                SortMultiTracker::new(
                    5,   // max_age
                    3,   // min_hits
                    0.3, // iou_threshold
                    0.5, // init_tracker_min_score
                    [1.0, 1.0, 10.0, 10.0],
                    [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                )
            },
            |mut tracker| {
                for det_frame in &detections {
                    let _result = tracker
                        .update(black_box(det_frame.view()), false, false)
                        .unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_bytetrack_update(c: &mut Criterion) {
    let detections = create_test_detections(20, 10);

    c.bench_function("bytetrack_update_20_detections", |b| {
        b.iter_batched(
            || {
                ByteMultiTracker::new(
                    5,   // max_age
                    3,   // min_hits
                    0.3, // iou_threshold
                    0.5, // init_tracker_min_score
                    0.6, // high_score_threshold
                    0.1, // low_score_threshold
                    [1.0, 1.0, 10.0, 10.0],
                    [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                )
            },
            |mut tracker| {
                for det_frame in &detections {
                    let _result = tracker
                        .update(black_box(det_frame.view()), false, false)
                        .unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_iou_calculation(c: &mut Criterion) {
    let detections = Array2::from_shape_vec((50, 4), (0..200).map(|i| i as f32).collect()).unwrap();

    let tracks =
        Array2::from_shape_vec((30, 4), (0..120).map(|i| (i as f32) + 0.5).collect()).unwrap();

    c.bench_function("iou_calculation_50x30", |b| {
        b.iter(|| ioutrack::bbox::ious(black_box(detections.view()), black_box(tracks.view())))
    });
}

fn bench_detection_splitting(c: &mut Criterion) {
    // Create a larger set of detections with mixed scores
    let mut data = Vec::with_capacity(100 * 5);
    for i in 0..100 {
        let score = if i % 3 == 0 {
            0.8
        } else if i % 3 == 1 {
            0.4
        } else {
            0.05
        };
        data.extend(&[i as f32, i as f32, (i + 50) as f32, (i + 30) as f32, score]);
    }
    let detections = Array2::from_shape_vec((100, 5), data).unwrap();

    c.bench_function("detection_splitting_100_detections", |b| {
        b.iter_batched(
            || {
                ByteMultiTracker::new(
                    5,
                    3,
                    0.3,
                    0.5,
                    0.6,
                    0.1,
                    [1.0, 1.0, 10.0, 10.0],
                    [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                )
            },
            |mut tracker| {
                let _result = tracker
                    .update(black_box(detections.view()), false, false)
                    .unwrap();
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_sort_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_various_detection_counts");

    for &n_detections in &[5, 10, 20, 50, 100] {
        let detections = create_test_detections(n_detections, 10);

        group.bench_with_input(
            BenchmarkId::new("detections", n_detections),
            &detections,
            |b, detections| {
                b.iter_batched(
                    || {
                        SortMultiTracker::new(
                            5,
                            3,
                            0.3,
                            0.5,
                            [1.0, 1.0, 10.0, 10.0],
                            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                        )
                    },
                    |mut tracker| {
                        for det_frame in detections {
                            let _result = tracker
                                .update(black_box(det_frame.view()), false, false)
                                .unwrap();
                        }
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_single_frame_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_frame_update");

    for &n_detections in &[10, 50, 100, 200] {
        let mut data = Vec::with_capacity(n_detections * 5);
        for i in 0..n_detections {
            data.extend(&[i as f32, i as f32, (i + 50) as f32, (i + 30) as f32, 0.8]);
        }
        let detection = Array2::from_shape_vec((n_detections, 5), data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("sort", n_detections),
            &detection,
            |b, detection| {
                b.iter_batched(
                    || {
                        SortMultiTracker::new(
                            5,
                            3,
                            0.3,
                            0.5,
                            [1.0, 1.0, 10.0, 10.0],
                            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                        )
                    },
                    |mut tracker| {
                        let _result = tracker
                            .update(black_box(detection.view()), false, false)
                            .unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bytetrack", n_detections),
            &detection,
            |b, detection| {
                b.iter_batched(
                    || {
                        ByteMultiTracker::new(
                            5,
                            3,
                            0.3,
                            0.5,
                            0.6,
                            0.1,
                            [1.0, 1.0, 10.0, 10.0],
                            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                        )
                    },
                    |mut tracker| {
                        let _result = tracker
                            .update(black_box(detection.view()), false, false)
                            .unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_iou_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("iou_optimizations");

    for &size in &[50, 100, 200, 500] {
        let detections =
            Array2::from_shape_vec((size, 4), (0..size * 4).map(|i| i as f32).collect()).unwrap();
        let tracks =
            Array2::from_shape_vec((size, 4), (0..size * 4).map(|i| (i as f32) + 0.5).collect())
                .unwrap();

        group.bench_with_input(
            BenchmarkId::new("original", size),
            &(&detections, &tracks),
            |b, (dets, trks)| {
                b.iter(|| ioutrack::bbox::ious(black_box(dets.view()), black_box(trks.view())))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("optimized", size),
            &(&detections, &tracks),
            |b, (dets, trks)| {
                b.iter(|| {
                    ioutrack::bbox::ious_optimized(black_box(dets.view()), black_box(trks.view()))
                })
            },
        );
    }
    group.finish();
}

fn bench_sparse_assignment(c: &mut Criterion) {
    use rand::Rng;
    let mut group = c.benchmark_group("assignment_algorithms");

    for &sparsity in &[10, 25, 50, 75] {
        // percentage of valid assignments
        let size = 100;
        let threshold = 0.5;
        let mut cost_matrix_data = vec![1.0f32; size * size]; // Initialize with high cost

        // Create sparse matrix
        let valid_count = (size * size * sparsity) / 100;
        let mut rng = rand::rng();
        for _ in 0..valid_count {
            let i = rng.random_range(0..size);
            let j = rng.random_range(0..size);
            let idx = i * size + j;
            cost_matrix_data[idx] = rng.random_range(0.0..threshold);
        }

        let cost_matrix = Array2::from_shape_vec((size, size), cost_matrix_data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("hungarian", sparsity),
            &cost_matrix,
            |b, matrix| {
                b.iter(|| {
                    ioutrack::hungarian::HungarianSolver::solve(black_box(matrix.view()), threshold)
                })
            },
        );
    }
    group.finish();
}

fn bench_spatial_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_indexing");

    for &n_objects in &[50, 100, 200, 500] {
        // Create scattered objects (low density)
        let scattered_dets: Vec<[f32; 4]> = (0..n_objects)
            .map(|i| {
                let x = (i as f32) * 100.0;
                let y = (i as f32) * 100.0;
                [x, y, x + 50.0, y + 50.0]
            })
            .collect();

        let scattered_tracks: Vec<[f32; 4]> = (0..n_objects)
            .map(|i| {
                let x = (i as f32) * 100.0 + 25.0;
                let y = (i as f32) * 100.0 + 25.0;
                [x, y, x + 50.0, y + 50.0]
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("spatial_query", n_objects),
            &(&scattered_dets, &scattered_tracks),
            |b, (dets, tracks)| {
                b.iter_batched(
                    || ioutrack::spatial::SpatialTracker::new(100.0),
                    |mut tracker| {
                        tracker.update(black_box(dets), black_box(tracks));
                        let _pairs = tracker.get_candidate_pairs();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Compare with brute force approach
        group.bench_with_input(
            BenchmarkId::new("brute_force_pairs", n_objects),
            &(&scattered_dets, &scattered_tracks),
            |b, (dets, tracks)| {
                b.iter(|| {
                    let mut pairs = Vec::new();
                    for i in 0..dets.len() {
                        for j in 0..tracks.len() {
                            pairs.push((i, j));
                        }
                    }
                    black_box(pairs)
                })
            },
        );

        // Test integrated spatial indexing in SORT tracker
        let mut detections_5d = Vec::new();
        for bbox in &scattered_dets {
            detections_5d.extend_from_slice(bbox);
            detections_5d.push(0.8); // confidence
        }
        let detections_array = Array2::from_shape_vec((n_objects, 5), detections_5d).unwrap();

        group.bench_with_input(
            BenchmarkId::new("sort_spatial_integrated", n_objects),
            &detections_array,
            |b, detection| {
                b.iter_batched(
                    || {
                        let mut tracker = SortMultiTracker::new(
                            5,
                            3,
                            0.3,
                            0.5,
                            [1.0, 1.0, 10.0, 10.0],
                            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                        );
                        tracker.set_spatial_threshold(25); // Force spatial indexing for 50+ objects
                        tracker
                    },
                    |mut tracker| {
                        let _result = tracker
                            .update(black_box(detection.view()), false, false)
                            .unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sort_standard_integrated", n_objects),
            &detections_array,
            |b, detection| {
                b.iter_batched(
                    || {
                        let mut tracker = SortMultiTracker::new(
                            5,
                            3,
                            0.3,
                            0.5,
                            [1.0, 1.0, 10.0, 10.0],
                            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
                        );
                        tracker.set_spatial_threshold(u32::MAX); // Disable spatial indexing
                        tracker
                    },
                    |mut tracker| {
                        let _result = tracker
                            .update(black_box(detection.view()), false, false)
                            .unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sort_update,
    bench_bytetrack_update,
    bench_sort_various_sizes,
    bench_single_frame_update,
    bench_iou_calculation,
    bench_detection_splitting,
    bench_iou_optimizations,
    bench_sparse_assignment,
    bench_spatial_indexing
);
criterion_main!(benches);
