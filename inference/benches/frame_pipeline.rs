use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::RgbImage;
use military_target_detector::{
    frame_pipeline::TileDetection, image_utils::calculate_scale_factors,
};
use rand::Rng;

/// Create test frame data
fn create_test_image(width: u32, height: u32) -> RgbImage {
    let mut image = RgbImage::new(width, height);

    // Add some pattern for variety
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let r = ((x) % 255) as u8;
        let g = ((y) % 255) as u8;
        let b = ((x.wrapping_mul(y)) % 255) as u8;
        *pixel = image::Rgb([r, g, b]);
    }

    image
}

/// Generate test detections
fn generate_test_detections(count: usize) -> Vec<TileDetection> {
    let mut rng = rand::thread_rng();
    let mut detections = Vec::with_capacity(count);

    for i in 0..count {
        detections.push(TileDetection {
            x: rng.gen_range(0.0..1800.0),
            y: rng.gen_range(0.0..1000.0),
            w: rng.gen_range(20.0..200.0),
            h: rng.gen_range(20.0..200.0),
            confidence: rng.gen_range(0.3..0.95),
            class_id: rng.gen_range(0..8),
            class_name: format!("class_{}", i % 8),
            tile_idx: 0,
            vx: Some(rng.gen_range(-5.0..5.0)),
            vy: Some(rng.gen_range(-5.0..5.0)),
            track_id: Some(i as u32),
        });
    }

    detections
}

/// Benchmark image preprocessing operations
fn bench_image_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_preprocessing");

    let image_sizes = [(640, 480), (1280, 720), (1920, 1080)];

    for (width, height) in image_sizes.iter() {
        let image = create_test_image(*width, *height);

        group.throughput(Throughput::Bytes((*width * *height * 3) as u64));
        group.bench_with_input(
            BenchmarkId::new("create_image", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, (width, height)| {
                b.iter(|| create_test_image(*width, *height));
            },
        );

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("scale_calculation", format!("{}x{}", width, height)),
            &image,
            |b, _image| {
                b.iter(|| calculate_scale_factors(*width, *height, 640.0));
            },
        );
    }

    group.finish();
}

/// Benchmark detection processing operations
fn bench_detection_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_processing");

    let detection_counts = [10, 25, 50, 100];

    for &count in detection_counts.iter() {
        let detections = generate_test_detections(count);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("filter_high_confidence", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let _filtered: Vec<&TileDetection> =
                        detections.iter().filter(|d| d.confidence > 0.7).collect();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sort_by_confidence", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let mut sorted = detections.clone();
                    sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("extract_track_ids", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let _track_ids: Vec<u32> =
                        detections.iter().filter_map(|d| d.track_id).collect();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");

    let data_sizes = [100, 500, 1000, 2000];

    for &size in data_sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("vector_allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let _vec: Vec<TileDetection> = Vec::with_capacity(size);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("detection_cloning", size),
            &size,
            |b, &size| {
                let detections = generate_test_detections(size);
                b.iter(|| {
                    let _cloned = detections.clone();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark frame rate simulation
fn bench_frame_rate_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_rate_simulation");

    let fps_targets = [15, 30, 60];
    let detection_count = 25;

    for &fps in fps_targets.iter() {
        let frame_interval = std::time::Duration::from_micros(1_000_000 / fps);
        let detections = generate_test_detections(detection_count);

        group.throughput(Throughput::Elements(fps as u64));
        group.bench_with_input(
            BenchmarkId::new("process_at_fps", fps),
            &detections,
            |b, detections| {
                b.iter(|| {
                    // Simulate processing multiple frames at target FPS
                    for _ in 0..5 {
                        let _processed: Vec<&TileDetection> =
                            detections.iter().filter(|d| d.confidence > 0.5).collect();

                        // Simulate frame interval (in benchmark context)
                        let _interval = frame_interval;
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    pipeline_benches,
    bench_image_preprocessing,
    bench_detection_processing,
    bench_memory_operations,
    bench_frame_rate_simulation,
);

criterion_main!(pipeline_benches);
