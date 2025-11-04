/// Single image detection example with ONNX Runtime + GPU (YOLOv8m - Full/Accurate)
///
/// This uses the full model which provides:
/// - Accuracy: Best detection quality
/// - Speed: ~490ms (slower, but most accurate)
/// - Size: 99MB
///
/// Usage:
///   cargo run --release --features metal --example detect_full [image_path]
use military_target_detector::types::{DetectorConfig, ImageData};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let image_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "test_data/test_tank.jpg".to_string()
    };

    println!("🎯 Military Target Detector - GPU Accelerated (YOLOv8m - Full)\n");
    println!("⚙️  Backend: ONNX Runtime (CoreML/Metal on macOS)\n");

    // Configuration for YOLOv8m (medium/accurate model)
    let config = DetectorConfig {
        model_path: "../models/military_target_detector.onnx".to_string(),
        confidence_threshold: 0.25,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: true,
        ..Default::default()
    };

    // Load image
    print!("📷 Loading image... ");
    let image = ImageData::from_file(&image_path)?;
    println!("✓ ({}x{})", image.width, image.height);

    // Load model
    print!("📦 Loading model on GPU... ");
    let start = Instant::now();
    let mut detector = military_target_detector::MilitaryTargetDetector::new(config)?;
    println!("✓ ({:.2}s)", start.elapsed().as_secs_f32());

    // Run inference
    print!("🚀 Running GPU inference... ");
    let start = Instant::now();
    let detections = detector.detect(&image)?;
    let inference_time = start.elapsed();
    println!("✓ ({:.0}ms)", inference_time.as_secs_f32() * 1000.0);

    // Print results
    println!("\n📊 Results:");
    println!("  Detections: {}", detections.len());
    println!(
        "  Inference: {:.0}ms ({:.1} FPS)",
        inference_time.as_secs_f32() * 1000.0,
        1.0 / inference_time.as_secs_f32()
    );

    if detections.is_empty() {
        println!("\n  ℹ️  No targets detected");
        return Ok(());
    }

    println!("\n🎯 Detected Targets:");
    for (i, det) in detections.iter().enumerate() {
        println!("\n  Detection #{}:", i + 1);
        println!("    Class: {:?}", det.class);
        println!("    Confidence: {:.1}%", det.confidence * 100.0);
        println!(
            "    Box: x={:.3}, y={:.3}, w={:.3}, h={:.3}",
            det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
        );
    }

    Ok(())
}
