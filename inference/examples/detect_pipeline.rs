/// Detection Pipeline Example
///
/// Demonstrates single-image pipeline with shared detector pool:
/// 1. Pre-processing: Tile extraction + shadow removal
/// 2. Execution: Batch detection via shared detector pool
/// 3. Post-processing: NMS + nested detection filtering + coordinate merging
///
/// The detector pool (with its model session) is shared and can be reused
/// across multiple pipeline executions.
///
/// Usage:
///   cargo run --release --features metal --example detect_pipeline <image_path> [output_path]
use image::ImageReader;
use military_target_detector::batch_executor::BatchConfig;
use military_target_detector::detector_pool::DetectorPool;
use military_target_detector::image_utils::{draw_rect, draw_text, generate_class_color};
use military_target_detector::pipeline::{DetectionPipeline, PipelineConfig};
use military_target_detector::types::DetectorConfig;
use std::env;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🎯 YOLO Detection Pipeline - Single Image Processing\n");

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
        "output_pipeline_annotated.jpg".to_string()
    };

    let start_total = Instant::now();

    // Load image
    print!("📷 Loading image... ");
    let original_img = ImageReader::open(&image_path)?.decode()?.to_rgb8();
    let img_width = original_img.width();
    let img_height = original_img.height();
    println!("✓ ({}x{})", img_width, img_height);

    // Create SHARED detector pool (can be reused for many images)
    print!("📦 Creating shared detector pool... ");
    let start_load = Instant::now();

    let num_workers = 2;
    let batch_config = BatchConfig {
        batch_size: 4,
        timeout_ms: 50,
    };

    let detector_config = DetectorConfig {
        model_path: "../models/yolov8m_batch_fp16.onnx".to_string(),
        confidence_threshold: 0.22,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: true,
        ..Default::default()
    };

    let detector_pool = Arc::new(DetectorPool::new(
        num_workers,
        detector_config,
        batch_config,
    )?);
    println!("✓ ({:.2}s)", start_load.elapsed().as_secs_f32());

    println!("\n💡 Note: Detector pool is shared and can process multiple images");
    println!("   Each pipeline execution reuses the same model session\n");

    // Create pipeline with shared detector pool
    println!("🔧 Building detection pipeline:");
    println!("  • Stage 1: Preprocess (tile extraction + shadow removal)");
    println!("  • Stage 2: Execution (batch detection via shared pool)");
    println!("  • Stage 3: Postprocess (NMS + nested filtering + coordinate merging)");

    let pipeline_config = PipelineConfig {
        tile_size: 640,
        overlap: 64,
        allowed_classes: vec![2, 3, 4, 7], // car, motorcycle, airplane, truck
        iou_threshold: 0.5,
    };

    let pipeline = DetectionPipeline::new(Arc::clone(&detector_pool), pipeline_config);
    println!("  ✓ Pipeline ready\n");

    // Process single image through pipeline
    println!("🚀 Processing image through pipeline...");
    let (result, timing) = pipeline.process_with_timing(&original_img)?;

    // Print timing
    println!("\n  ⏱️  Stage Timing:");
    println!("    • Preprocess: {:.1}ms", timing.preprocess_ms);
    println!("    • Execution: {:.1}ms", timing.execution_ms);
    println!("    • Postprocess: {:.1}ms", timing.postprocess_ms);
    println!("    • Total: {:.1}ms", timing.total_ms);

    // Print results
    println!("\n📊 Detection Results:");
    let total_initial = result.detections.len() + result.duplicates_removed + result.nested_removed;
    println!("  • Total initial detections: {}", total_initial);
    println!(
        "  • Duplicates removed (NMS): {}",
        result.duplicates_removed
    );
    println!("  • Nested detections removed: {}", result.nested_removed);
    println!("  • Final detections: {}", result.detections.len());

    if !result.detections.is_empty() {
        println!("\n🎯 Detected Targets:\n");
        for (i, det) in result.detections.iter().enumerate() {
            println!("  Detection #{}:", i + 1);
            println!("    Class: {} (ID: {})", det.class_name, det.class_id);
            println!("    Confidence: {:.1}%", det.confidence * 100.0);
            println!(
                "    Box: x={:.3}, y={:.3}, w={:.3}, h={:.3}",
                det.x / img_width as f32,
                det.y / img_height as f32,
                det.w / img_width as f32,
                det.h / img_height as f32
            );
            println!("    Tile: {}", det.tile_idx);
            println!();
        }
    }

    // Draw results on original image
    print!("🎨 Drawing bounding boxes... ");
    let mut annotated_img = original_img;

    for det in &result.detections {
        let color = generate_class_color(det.class_id);

        draw_rect(
            &mut annotated_img,
            det.x as i32,
            det.y as i32,
            det.w as u32,
            det.h as u32,
            color,
            2,
        );

        let label = format!("{} {:.0}%", det.class_name, det.confidence * 100.0);
        let padding = 3;
        let text_x = (det.x as i32 + padding).max(0);
        let text_y = (det.y as i32 + padding).max(0);

        if det.h > 20.0 && det.w > 30.0 {
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

    println!("\n💡 The detector pool can now be reused for more images!");
    println!("   Example: pipeline.process(&another_image)?;");

    Ok(())
}
