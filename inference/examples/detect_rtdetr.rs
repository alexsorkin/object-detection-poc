/// RT-DETR Detection Example
///
/// Demonstrates RT-DETR (Real-time DEtection TRansformer) detection:
/// - No NMS required (transformer outputs unique detections)
/// - Better accuracy for small objects
/// - 300 queries vs 8400 YOLO anchors
///
/// Usage:
///   cargo run --release --features metal --example detect_rtdetr <image_path> [output_path]

use image::ImageReader;
use military_target_detector::image_utils::{draw_rect, draw_text, generate_class_color};
use military_target_detector::types::DetectorConfig;
use military_target_detector::RTDETRDetector;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🎯 RT-DETR Detection - Single Image Processing\n");

    // Parse arguments
    let args: Vec<String> = env::args().collect();
    let image_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "test_data/yolo_airport.jpg".to_string()
    };

    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        "output_rtdetr_annotated.jpg".to_string()
    };

    let start_total = Instant::now();

    // Load image
    print!("📷 Loading image... ");
    let original_img = ImageReader::open(&image_path)?.decode()?.to_rgb8();
    let img_width = original_img.width();
    let img_height = original_img.height();
    println!("✓ ({}x{})", img_width, img_height);

    // Platform-specific model selection
    #[cfg(any(feature = "cuda", feature = "tensorrt"))]
    let (fp16_path, fp32_path) = (
        "../models/rf-detr-medium.onnx",  // Use same model (it's FP32)
        "../models/rf-detr-medium.onnx",
    );

    #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "tensorrt"))))]
    let (fp16_path, fp32_path) = (
        "../models/rf-detr-medium.onnx",
        "../models/rf-detr-medium.onnx",
    );

    #[cfg(not(any(feature = "metal", feature = "cuda", feature = "tensorrt")))]
    let (fp16_path, fp32_path) = (
        "../models/rf-detr-medium.onnx",
        "../models/rf-detr-medium.onnx",
    );

    // Create detector configuration
    print!("📦 Creating RT-DETR detector... ");
    let start_load = Instant::now();

    let detector_config = DetectorConfig {
        fp16_model_path: Some(fp16_path.to_string()),
        fp32_model_path: Some(fp32_path.to_string()),
        confidence_threshold: 0.3,  // RT-DETR can use lower threshold (no NMS noise)
        nms_threshold: 0.0,         // Not used by RT-DETR
        input_size: (576, 576),     // RT-DETR uses 576x576
        use_gpu: true,
        ..Default::default()
    };

    let mut detector = RTDETRDetector::new(detector_config)?;
    println!("✓ ({:.2}s)", start_load.elapsed().as_secs_f32());

    println!("\n💡 RT-DETR Advantages:");
    println!("   • No NMS required (transformer outputs unique detections)");
    println!("   • 300 queries vs 8400 YOLO anchors (13x fewer boxes to process)");
    println!("   • Better accuracy for small and overlapping objects");
    println!("   • End-to-end detection without post-processing\n");

    // Convert image to ImageData format
    let image_data = military_target_detector::types::ImageData {
        data: original_img.as_raw().clone(),
        width: img_width,
        height: img_height,
        format: military_target_detector::types::ImageFormat::RGB,
    };

    // Run detection
    println!("🚀 Running RT-DETR detection...");
    let start_detect = Instant::now();
    let detections = detector.detect(&image_data)?;
    let detect_time = start_detect.elapsed();

    println!("\n  ⏱️  Detection Time: {:.1}ms", detect_time.as_secs_f32() * 1000.0);

    // Print results
    println!("\n📊 Detection Results:");
    println!("  • Total detections: {}", detections.len());

    if !detections.is_empty() {
        println!("\n🎯 Detected Objects:\n");
        for (i, det) in detections.iter().enumerate() {
            println!("  Detection #{}:", i + 1);
            println!("    Class: {} (ID: {:?})", det.class.name(), det.class);
            println!("    Confidence: {:.1}%", det.confidence * 100.0);
            println!(
                "    Box: x={:.3}, y={:.3}, w={:.3}, h={:.3}",
                det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
            );
            println!();
        }
    }

    // Draw results on original image
    print!("🎨 Drawing bounding boxes... ");
    let mut annotated_img = original_img.clone();

    for det in &detections {
        // Get class ID for coloring
        let class_id = match det.class {
            military_target_detector::types::TargetClass::Class(id) => id,
            _ => 0,
        };
        
        let color = generate_class_color(class_id);

        // Convert normalized coordinates to pixels
        let x = (det.bbox.x * img_width as f32) as i32;
        let y = (det.bbox.y * img_height as f32) as i32;
        let w = (det.bbox.width * img_width as f32) as u32;
        let h = (det.bbox.height * img_height as f32) as u32;

        draw_rect(&mut annotated_img, x, y, w, h, color, 2);

        let label = format!("{} {:.0}%", det.class.name(), det.confidence * 100.0);
        let padding = 3;
        let text_x = (x + padding).max(0);
        let text_y = (y + padding).max(0);

        if h > 20 && w > 30 {
            draw_text(
                &mut annotated_img,
                &label,
                text_x,
                text_y,
                image::Rgb([255, 255, 255]),
                None,
            );
        }
    }
    println!("✓");

    // Save
    annotated_img.save(&output_path)?;
    println!("💾 Saved to: {}", output_path);

    println!(
        "\n⏱️  Total time: {:.2}s",
        start_total.elapsed().as_secs_f32()
    );

    println!("\n📈 Performance Comparison:");
    println!("   YOLO:    ~8400 anchor boxes → NMS required");
    println!("   RT-DETR: ~300 queries → No NMS needed ✨");

    Ok(())
}
