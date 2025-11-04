/// Example: Single image detection using YOLOv8s (Mini - Fast)
/// 
/// This uses the mini model which provides:
/// - Speed: ~187ms (fast)
/// - Accuracy: Good for most use cases
/// - Size: 43MB (compact)
///
/// Usage:
///   cargo run --release --features metal --example detect_mini <image_path>

use military_target_detector::{DetectorConfig, ImageData, MilitaryTargetDetector};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let image_path = if args.len() > 1 {
        &args[1]
    } else {
        "test_data/test_tank.jpg"
    };

    println!("🎯 YOLOv8s (Mini) - Military Target Detector");
    println!("═══════════════════════════════════════════\n");

    println!("📦 Loading model (Mini - Fast)...");
    let load_start = Instant::now();
    
    let config = DetectorConfig {
        model_path: "../models/military_target_detector_mini.onnx".to_string(),
        ..Default::default()
    };

    let mut detector = MilitaryTargetDetector::new(config)
        .expect("Failed to create detector");
    
    println!("✓ Model loaded ({:.2}s)\n", load_start.elapsed().as_secs_f32());

    println!("📷 Loading image: {}", image_path);
    let image = ImageData::from_file(image_path)
        .expect("Failed to load image");
    
    println!("✓ Image loaded: {}x{}\n", image.width, image.height);

    println!("🔍 Running detection...");
    let detect_start = Instant::now();
    
    let detections = detector.detect(&image)
        .expect("Detection failed");
    
    let detect_time = detect_start.elapsed();
    
    println!("✓ Detection complete ({:.1}ms)\n", detect_time.as_secs_f32() * 1000.0);

    if detections.is_empty() {
        println!("ℹ️  No targets detected");
    } else {
        println!("🎯 Detected {} target(s):\n", detections.len());
        
        for (i, det) in detections.iter().enumerate() {
            println!("  Target {}:", i + 1);
            println!("    Class: {:?}", det.class);
            println!("    Confidence: {:.1}%", det.confidence * 100.0);
            println!("    Position: ({:.0}, {:.0})", det.bbox.x, det.bbox.y);
            println!("    Size: {:.0}x{:.0}", det.bbox.width, det.bbox.height);
            println!();
        }
    }

    println!("📊 Performance:");
    println!("  Model: YOLOv8s (Mini - 43MB)");
    println!("  Inference: {:.1}ms", detect_time.as_secs_f32() * 1000.0);
    println!("  FPS: {:.1}", 1.0 / detect_time.as_secs_f32());
    println!("\n💡 This is the fast model - recommended for real-time use!");
}
