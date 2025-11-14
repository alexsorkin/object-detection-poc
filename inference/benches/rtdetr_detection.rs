use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::{Rgb, RgbImage};
use military_target_detector::{
    detector_rtdetr::RTDETRDetector,
    types::{DetectorConfig, ImageData, ImageFormat, RTDETRModel},
};
use std::env;

/// Create test image data with realistic resolution
fn create_test_image(width: u32, height: u32) -> RgbImage {
    let mut image = RgbImage::new(width, height);

    // Add some realistic pattern to simulate actual image content
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let r = ((x as f32 / width as f32) * 255.0) as u8;
        let g = ((y as f32 / height as f32) * 255.0) as u8;
        let b = ((x.wrapping_mul(y)) % 255) as u8;
        *pixel = Rgb([r, g, b]);
    }
    image
}

/// Benchmark RT-DETR detection performance across different image sizes
fn bench_rtdetr_detection(c: &mut Criterion) {
    // Skip if no model available
    let model_dir = env::var("DEFENITY_MODEL_DIR").unwrap_or_else(|_| "../models".to_string());
    let model_path = format!("{}/{}", model_dir, RTDETRModel::R18VD_FP32.filename());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!(
            "⚠️  Skipping detection benchmarks - model not found: {}",
            model_path
        );
        return;
    }

    let detector_config = DetectorConfig {
        model_path,
        confidence_threshold: 0.5,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: false, // Use CPU for consistent benchmarking
        ..Default::default()
    };

    let mut group = c.benchmark_group("rtdetr_detection");

    // Test different image sizes commonly used in surveillance/military applications
    let test_sizes = [
        (640, 480),   // VGA
        (1280, 720),  // HD
        (1920, 1080), // Full HD
    ];

    for (width, height) in test_sizes.iter() {
        let image = create_test_image(*width, *height);
        let image_data = ImageData {
            data: image.into_raw(),
            width: *width,
            height: *height,
            format: ImageFormat::RGB,
        };

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("detect", format!("{}x{}", width, height)),
            &image_data,
            |b, image_data| {
                b.iter(|| {
                    // Create a new detector for each iteration due to &mut self requirement
                    let mut detector_for_bench = RTDETRDetector::new(detector_config.clone())
                        .expect("Failed to create RT-DETR detector");
                    detector_for_bench
                        .detect(image_data)
                        .expect("Detection failed")
                });
            },
        );
    }
    group.finish();
}

/// Benchmark batch detection performance
fn bench_rtdetr_batch_detection(c: &mut Criterion) {
    let model_dir = env::var("DEFENITY_MODEL_DIR").unwrap_or_else(|_| "../models".to_string());
    let model_path = format!("{}/{}", model_dir, RTDETRModel::R18VD_FP32.filename());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!(
            "⚠️  Skipping batch detection benchmarks - model not found: {}",
            model_path
        );
        return;
    }

    let detector_config = DetectorConfig {
        model_path,
        confidence_threshold: 0.5,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: false,
        ..Default::default()
    };

    let mut group = c.benchmark_group("rtdetr_batch_detection");

    // Test batch sizes commonly used in video processing
    let batch_sizes = [1, 2, 4];
    let image_size = (1280, 720); // HD resolution

    for batch_size in batch_sizes.iter() {
        let images: Vec<ImageData> = (0..*batch_size)
            .map(|_| {
                let image = create_test_image(image_size.0, image_size.1);
                ImageData {
                    data: image.into_raw(),
                    width: image_size.0,
                    height: image_size.1,
                    format: ImageFormat::RGB,
                }
            })
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_detect", batch_size),
            &images,
            |b, images| {
                b.iter(|| {
                    let mut detector = RTDETRDetector::new(detector_config.clone())
                        .expect("Failed to create RT-DETR detector");
                    for image in images.iter() {
                        detector.detect(image).expect("Detection failed");
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark confidence threshold impact
fn bench_confidence_thresholds(c: &mut Criterion) {
    let model_dir = env::var("DEFENITY_MODEL_DIR").unwrap_or_else(|_| "../models".to_string());
    let model_path = format!("{}/{}", model_dir, RTDETRModel::R18VD_FP32.filename());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!(
            "⚠️  Skipping confidence threshold benchmarks - model not found: {}",
            model_path
        );
        return;
    }

    let mut group = c.benchmark_group("confidence_thresholds");
    let image = create_test_image(1280, 720);
    let image_data = ImageData {
        data: image.into_raw(),
        width: 1280,
        height: 720,
        format: ImageFormat::RGB,
    };

    let confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9];

    for &threshold in confidence_thresholds.iter() {
        let detector_config = DetectorConfig {
            model_path: model_path.clone(),
            confidence_threshold: threshold,
            nms_threshold: 0.45,
            input_size: (640, 640),
            use_gpu: false,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("confidence", format!("{:.1}", threshold)),
            &image_data,
            |b, image_data| {
                b.iter(|| {
                    let mut detector = RTDETRDetector::new(detector_config.clone())
                        .expect("Failed to create RT-DETR detector");
                    detector.detect(image_data).expect("Detection failed")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    detection_benches,
    bench_rtdetr_detection,
    bench_rtdetr_batch_detection,
    bench_confidence_thresholds,
);

criterion_main!(detection_benches);
