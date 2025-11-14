use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::{Rgb, RgbImage};
use military_target_detector::{
    frame_pipeline::TileDetection,
    image_utils::{calculate_scale_factors, draw_rect_batch, prepare_annotation_data},
};
use rand::Rng;

/// Generate test detection data for image processing benchmarks
fn generate_test_tile_detections(count: usize) -> Vec<TileDetection> {
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
            class_name: format!("class_{}", rng.gen_range(0..8)),
            tile_idx: 0,
            vx: None,
            vy: None,
            track_id: Some(i as u32),
        });
    }

    detections
}

/// Create test image with realistic content
fn create_test_image(width: u32, height: u32) -> RgbImage {
    let mut image = RgbImage::new(width, height);

    // Create a gradient pattern to simulate realistic image content
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let r = ((x as f32 / width as f32) * 255.0) as u8;
        let g = ((y as f32 / height as f32) * 255.0) as u8;
        let b = ((x.wrapping_mul(y)) % 255) as u8;
        *pixel = Rgb([r, g, b]);
    }

    // Add some noise for realism
    let mut rng = rand::thread_rng();
    for _ in 0..(width * height / 100) {
        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);
        let noise = Rgb([
            rng.gen_range(0..255),
            rng.gen_range(0..255),
            rng.gen_range(0..255),
        ]);
        image.put_pixel(x, y, noise);
    }

    image
}

/// Benchmark scale factor calculation
fn bench_scale_factors(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_factors");

    let test_sizes = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)];

    let target_sizes = [320.0, 640.0, 1024.0];

    for (width, height) in test_sizes.iter() {
        for &target_size in target_sizes.iter() {
            group.throughput(Throughput::Elements(1));
            group.bench_with_input(
                BenchmarkId::new(
                    "calculate_scale",
                    format!("{}x{}_target_{}", width, height, target_size as u32),
                ),
                &(*width, *height, target_size),
                |b, (width, height, target_size)| {
                    b.iter(|| calculate_scale_factors(*width, *height, *target_size));
                },
            );
        }
    }

    group.finish();
}

/// Benchmark annotation data preparation
fn bench_prepare_annotation_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("prepare_annotation_data");

    let detection_counts = [1, 5, 10, 25, 50, 100];
    let scale_factors = (1.5, 1.2); // Realistic scale factors

    for &count in detection_counts.iter() {
        let detections = generate_test_tile_detections(count);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("prepare_annotations", count),
            &detections,
            |b, detections| {
                b.iter(|| prepare_annotation_data(detections, scale_factors.0, scale_factors.1));
            },
        );
    }

    group.finish();
}

/// Benchmark batch rectangle drawing
fn bench_draw_rect_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_rect_batch");

    let image_sizes = [(1280, 720), (1920, 1080)];
    let rect_counts = [1, 5, 10, 25, 50];

    for (width, height) in image_sizes.iter() {
        for &count in rect_counts.iter() {
            let image = create_test_image(*width, *height);

            // Generate rectangles to draw
            let mut rng = rand::thread_rng();
            let rects: Vec<_> = (0..count)
                .map(|_| {
                    let x = rng.gen_range(0..(*width as i32 - 100));
                    let y = rng.gen_range(0..(*height as i32 - 100));
                    let w = rng.gen_range(20..100);
                    let h = rng.gen_range(20..100);
                    let color = Rgb([
                        rng.gen_range(0..255),
                        rng.gen_range(0..255),
                        rng.gen_range(0..255),
                    ]);
                    (x, y, w, h, color, 2) // thickness = 2
                })
                .collect();

            group.throughput(Throughput::Elements(count as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    "draw_rects",
                    format!("{}x{}_{}_rects", width, height, count),
                ),
                &(image.clone(), rects),
                |b, (base_image, rects)| {
                    b.iter(|| {
                        let mut img = base_image.clone();
                        draw_rect_batch(&mut img, rects);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark complete annotation pipeline
fn bench_complete_annotation_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_annotation_pipeline");

    let scenarios = [
        ("light_load", 10, (1280, 720)),
        ("medium_load", 25, (1920, 1080)),
        ("heavy_load", 50, (1920, 1080)),
    ];

    for (scenario_name, detection_count, (width, height)) in scenarios.iter() {
        let base_image = create_test_image(*width, *height);
        let detections = generate_test_tile_detections(*detection_count);
        let scale_factors = calculate_scale_factors(*width, *height, 640.0);

        group.throughput(Throughput::Elements(*detection_count as u64));
        group.bench_with_input(
            BenchmarkId::new("full_annotation", scenario_name),
            &(base_image, detections, scale_factors),
            |b, (base_image, detections, (scale_x, scale_y))| {
                b.iter(|| {
                    let mut image = base_image.clone();

                    // Prepare annotation data
                    let annotation_data = prepare_annotation_data(detections, *scale_x, *scale_y);

                    // Prepare rectangles for batch drawing
                    let rects: Vec<_> = annotation_data
                        .iter()
                        .map(|(x, y, w, h, color, _label, _show_label)| (*x, *y, *w, *h, *color, 2))
                        .collect();

                    // Draw rectangles
                    draw_rect_batch(&mut image, &rects);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    image_processing_benches,
    bench_scale_factors,
    bench_prepare_annotation_data,
    bench_draw_rect_batch,
    bench_complete_annotation_pipeline,
);

criterion_main!(image_processing_benches);
