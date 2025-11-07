/// Performance benchmark for YOLOv8 detection with ONNX Runtime + GPU
///
/// Usage:
///   cargo run --release --features metal --example benchmark_new
use military_target_detector::types::{DetectorConfig, ImageData};
use military_target_detector::YoloV8Detector;
use std::time::Instant;

fn benchmark_model(
    name: &str,
    model_path: &str,
    image: &ImageData,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("Testing: {}", name);
    println!("═══════════════════════════════════════════════════════\n");

    let config = DetectorConfig {
        model_path: model_path.to_string(),
        confidence_threshold: 0.25,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: true,
        ..Default::default()
    };

    // Load model
    print!("📦 Loading model... ");
    let start = Instant::now();
    let mut detector = YoloV8Detector::new(config)?;
    println!("✓ ({:.2}s)", start.elapsed().as_secs_f32());

    // Warmup runs (CoreML/Metal needs compilation time)
    print!("🔥 Warming up (10 runs)... ");
    let warmup_start = Instant::now();
    for _ in 0..10 {
        let _ = detector.detect(image)?;
    }
    let warmup_time = warmup_start.elapsed();
    println!(
        "✓ ({:.2}s total, {:.0}ms avg)",
        warmup_time.as_secs_f32(),
        warmup_time.as_millis() as f32 / 10.0
    );

    // Benchmark runs
    print!("⚡ Benchmarking (100 runs)... ");
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        let _detections = detector.detect(image)?;
        times.push(start.elapsed().as_secs_f32() * 1000.0);
    }
    println!("✓");

    // Calculate statistics
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let max = times[times.len() - 1];
    let avg = times.iter().sum::<f32>() / times.len() as f32;
    let p50 = times[times.len() / 2];
    let p95 = times[(times.len() as f32 * 0.95) as usize];
    let p99 = times[(times.len() as f32 * 0.99) as usize];

    println!("\n📊 Results:");
    println!("  Average:  {:.1}ms ({:.1} FPS)", avg, 1000.0 / avg);
    println!("  Median:   {:.1}ms", p50);
    println!("  Min:      {:.1}ms", min);
    println!("  Max:      {:.1}ms", max);
    println!("  P95:      {:.1}ms", p95);
    println!("  P99:      {:.1}ms", p99);
    println!();

    Ok(())
}

fn main() {
    println!("🔥 Military Target Detector Benchmark");
    println!("═══════════════════════════════════════════════════════\n");

    // Load test image once
    let image = ImageData::from_file("test_data/test_tank.jpg").expect("Failed to load test image");

    println!("✓ Test image loaded: {}x{}\n", image.width, image.height);

    // Test both models
    benchmark_model(
        "YOLOv8s (Mini - Fast) ⭐",
        "../models/military_target_detector_mini.onnx",
        &image,
    )
    .expect("Benchmark failed");

    println!("\n");

    benchmark_model(
        "YOLOv8m (Full - Accurate)",
        "../models/military_target_detector.onnx",
        &image,
    )
    .expect("Benchmark failed");

    println!("\n✅ Benchmark complete!");
    println!("\n💡 Recommendation: Use Mini model for real-time performance!");
}
