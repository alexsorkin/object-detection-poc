use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use military_target_detector::{
    frame_pipeline::TileDetection,
    tracking_utils::{calculate_iou, BoundingBox},
};
use rand::Rng;
use std::sync::Arc;

/// Generate realistic test detections for tracking benchmarks
fn generate_test_detections(
    count: usize,
    frame_width: f32,
    frame_height: f32,
) -> Vec<TileDetection> {
    let mut rng = rand::rng();
    let mut detections = Vec::with_capacity(count);

    for _i in 0..count {
        let width = rng.random_range(20.0..100.0);
        let height = rng.random_range(20.0..100.0);
        let x = rng.random_range(0.0..(frame_width - width));
        let y = rng.random_range(0.0..(frame_height - height));
        let confidence = rng.random_range(0.3..0.95);

        detections.push(TileDetection {
            x,
            y,
            w: width,
            h: height,
            confidence,
            class_id: rng.random_range(0..8), // Common COCO classes
            class_name: Arc::from(format!("class_{}", rng.random_range(0..8)).as_str()),
            tile_idx: 0,
            vx: None,
            vy: None,
            track_id: None,
        });
    }

    detections
}

/// Convert TileDetection to BoundingBox
fn tile_detection_to_bbox(detection: &TileDetection) -> BoundingBox {
    BoundingBox {
        x1: detection.x,
        y1: detection.y,
        x2: detection.x + detection.w,
        y2: detection.y + detection.h,
    }
}

/// Benchmark IoU calculation performance
fn bench_iou_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("iou_calculation");

    let detection_counts = [10, 25, 50, 100];
    let frame_size = (1920.0, 1080.0);

    for &count in detection_counts.iter() {
        let detections = generate_test_detections(count, frame_size.0, frame_size.1);

        // Create pairs for IoU calculation
        let pairs: Vec<_> = detections
            .iter()
            .take(count / 2)
            .zip(detections.iter().skip(count / 2))
            .collect();

        group.throughput(Throughput::Elements(pairs.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("calculate_iou", count),
            &pairs,
            |b, pairs| {
                b.iter(|| {
                    for (det1, det2) in pairs.iter() {
                        let bbox1 = tile_detection_to_bbox(det1);
                        let bbox2 = tile_detection_to_bbox(det2);
                        calculate_iou(&bbox1, &bbox2);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark detection filtering by confidence
fn bench_detection_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_filtering");

    let detection_counts = [50, 100, 200, 500];
    let frame_size = (1920.0, 1080.0);

    for &count in detection_counts.iter() {
        let detections = generate_test_detections(count, frame_size.0, frame_size.1);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("filter_by_confidence", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let _filtered: Vec<&TileDetection> =
                        detections.iter().filter(|d| d.confidence > 0.5).collect();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark detection sorting by confidence
fn bench_detection_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_sorting");

    let detection_counts = [50, 100, 200, 500];
    let frame_size = (1920.0, 1080.0);

    for &count in detection_counts.iter() {
        let detections = generate_test_detections(count, frame_size.0, frame_size.1);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("sort_by_confidence", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let mut sorted_detections = detections.clone();
                    sorted_detections
                        .sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark coordinate transformations
fn bench_coordinate_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinate_transforms");

    let detection_counts = [50, 100, 200];
    let scale_factors = [(1.0, 1.0), (0.5, 0.5), (2.0, 2.0)];
    let frame_size = (1920.0, 1080.0);

    for &count in detection_counts.iter() {
        let detections = generate_test_detections(count, frame_size.0, frame_size.1);

        for &(scale_x, scale_y) in scale_factors.iter() {
            group.throughput(Throughput::Elements(count as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    "transform_coordinates",
                    format!("{}_detections_scale_{}x{}", count, scale_x, scale_y),
                ),
                &detections,
                |b, detections| {
                    b.iter(|| {
                        let _transformed: Vec<_> = detections
                            .iter()
                            .map(|d| TileDetection {
                                x: d.x * scale_x,
                                y: d.y * scale_y,
                                w: d.w * scale_x,
                                h: d.h * scale_y,
                                confidence: d.confidence,
                                class_id: d.class_id,
                                class_name: Arc::clone(&d.class_name),
                                tile_idx: d.tile_idx,
                                vx: d.vx.map(|v| v * scale_x),
                                vy: d.vy.map(|v| v * scale_y),
                                track_id: d.track_id,
                            })
                            .collect();
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark detection area calculations
fn bench_detection_areas(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_areas");

    let detection_counts = [50, 100, 200, 500];
    let frame_size = (1920.0, 1080.0);

    for &count in detection_counts.iter() {
        let detections = generate_test_detections(count, frame_size.0, frame_size.1);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("calculate_areas", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let _areas: Vec<f32> = detections.iter().map(|d| d.w * d.h).collect();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    tracking_benches,
    bench_iou_calculation,
    bench_detection_filtering,
    bench_detection_sorting,
    bench_coordinate_transforms,
    bench_detection_areas,
);

criterion_main!(tracking_benches);
