//! Benchmarks for tracking algorithms

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ioutrack::{ByteTrack, Sort};
use ndarray::Array2;

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
        b.iter(|| {
            let mut tracker = Sort::new(
                5,   // max_age
                3,   // min_hits
                0.3, // iou_threshold
                0.5, // init_tracker_min_score
                [1.0, 1.0, 10.0, 10.0],
                [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
            );

            for det_frame in &detections {
                let _result = tracker.update(black_box(det_frame.view()), false, false);
            }
        })
    });
}

fn bench_bytetrack_update(c: &mut Criterion) {
    let detections = create_test_detections(20, 10);

    c.bench_function("bytetrack_update_20_detections", |b| {
        b.iter(|| {
            let mut tracker = ByteTrack::new(
                5,   // max_age
                3,   // min_hits
                0.3, // iou_threshold
                0.5, // init_tracker_min_score
                0.6, // high_score_threshold
                0.1, // low_score_threshold
                [1.0, 1.0, 10.0, 10.0],
                [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
            );

            for det_frame in &detections {
                let _result = tracker.update(black_box(det_frame.view()), false, false);
            }
        })
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
        let tracker = ByteTrack::new(
            5,
            3,
            0.3,
            0.5,
            0.6,
            0.1,
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        );

        b.iter(|| {
            // Access the private method through update to test the splitting performance
            let mut test_tracker = tracker.clone();
            let _result = test_tracker.update(black_box(detections.view()), false, false);
        })
    });
}

criterion_group!(
    benches,
    bench_sort_update,
    bench_bytetrack_update,
    bench_iou_calculation,
    bench_detection_splitting
);
criterion_main!(benches);
