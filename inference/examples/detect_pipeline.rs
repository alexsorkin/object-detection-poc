/// Detection Pipeline Example - Unified for YOLOv8 and RT-DETR
///
/// Demonstrates the complete detection pipeline with tiled batch processing:
/// 1. Pre-processing: Tile extraction (auto-sized 640×640) + shadow removal
/// 2. Execution: Parallel batch detection via batch executor
/// 3. Post-processing: NMS + nested detection filtering + coordinate merging
///
/// The batch executor is shared and can be reused across multiple images.
/// Frames are enqueued directly to the FrameExecutor which processes them one at a time.
///
/// Usage:
///   cargo run --release --features metal --example detect_pipeline [--detector yolov8|rtdetr] [--confidence %%] <image_path> [output_path]
///
/// Examples:
///   cargo run --release --features metal --example detect_pipeline
///   cargo run --release --features metal --example detect_pipeline -- --detector yolov8
///   cargo run --release --features metal --example detect_pipeline -- --detector rtdetr --confidence 35 test_data/my_image.jpg output.jpg
///   cargo run --release --features metal --example detect_pipeline -- --confidence 50
use image::ImageReader;
use military_target_detector::detector_trait::DetectorType;
use military_target_detector::frame_executor::{ExecutorConfig, FrameExecutor};
use military_target_detector::frame_pipeline::{DetectionPipeline, PipelineConfig};
use military_target_detector::image_utils::{draw_rect, draw_text, generate_class_color};
use military_target_detector::types::DetectorConfig;
use std::env;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse arguments
    let args: Vec<String> = env::args().collect();

    // Parse detector type
    let mut detector_type = DetectorType::RTDETR; // Default to RT-DETR
    let mut confidence_threshold = 0.50; // Default 50%
    let mut arg_idx = 1;

    if args.len() > arg_idx && args[arg_idx] == "--detector" {
        arg_idx += 1;
        if args.len() > arg_idx {
            detector_type = match args[arg_idx].to_lowercase().as_str() {
                "rtdetr" | "rt-detr" => DetectorType::RTDETR,
                "yolo" | "yolov8" => DetectorType::YOLOV8,
                _ => {
                    eprintln!("Invalid detector type. Use 'yolov8' or 'rtdetr'");
                    return Ok(());
                }
            };
            arg_idx += 1;
        }
    }

    // Parse confidence threshold
    if args.len() > arg_idx && args[arg_idx] == "--confidence" {
        arg_idx += 1;
        if args.len() > arg_idx {
            match args[arg_idx].parse::<f32>() {
                Ok(val) => {
                    confidence_threshold = (val / 100.0).clamp(0.0, 1.0); // Convert % to 0.0-1.0
                }
                Err(_) => {
                    eprintln!("Invalid confidence threshold. Use a number between 0-100");
                    return Ok(());
                }
            }
            arg_idx += 1;
        }
    }

    let detector_name = match detector_type {
        DetectorType::YOLOV8 => "YOLOv8",
        DetectorType::RTDETR => "RT-DETR",
    };

    println!(
        "🎯 {} Detection Pipeline - Single Image Processing\n",
        detector_name
    );

    let image_path = if args.len() > arg_idx {
        args[arg_idx].clone()
    } else {
        "test_data/yolo_airport.jpg".to_string()
    };

    let output_path = if args.len() > arg_idx + 1 {
        args[arg_idx + 1].clone()
    } else {
        format!(
            "output_{}.jpg",
            detector_name.to_lowercase().replace("-", "")
        )
    };

    let start_total = Instant::now();

    // Load image
    print!("📷 Loading image... ");
    let original_img = ImageReader::open(&image_path)?.decode()?.to_rgb8();
    let img_width = original_img.width();
    let img_height = original_img.height();
    println!("✓ ({}x{})", img_width, img_height);

    // Create SHARED frame executor (can be reused for many images)
    print!("📦 Creating frame executor... ");
    let start_load = Instant::now();

    let executor_config = ExecutorConfig {
        max_queue_depth: 10, // Allow more queued tiles for single image processing
    };
    let max_queue_depth = executor_config.max_queue_depth; // Save before move

    // Platform-specific model selection:
    // - NVIDIA (CUDA/TensorRT): Use FP16 for 2-3x speedup
    // - Apple (CoreML/Metal): Use FP32 (CoreML optimizes internally)
    // - CPU: Use FP32
    let (fp16_path, fp32_path) = match detector_type {
        DetectorType::YOLOV8 => {
            #[cfg(any(feature = "cuda", feature = "tensorrt"))]
            let paths = (
                "../models/yolov8m_batch_fp16.onnx",
                "../models/yolov8m_batch_fp32.onnx",
            );

            #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "tensorrt"))))]
            let paths = (
                "../models/yolov8m_batch_fp32.onnx",
                "../models/yolov8m_batch_fp32.onnx",
            );

            #[cfg(not(any(feature = "metal", feature = "cuda", feature = "tensorrt")))]
            let paths = (
                "../models/yolov8m_batch_fp32.onnx",
                "../models/yolov8m_batch_fp32.onnx",
            );

            paths
        }
        DetectorType::RTDETR => (
            "../models/rtdetr_v2_r18vd_batch.onnx",
            "../models/rtdetr_v2_r18vd_batch.onnx",
        ),
    };

    let (input_width, input_height) = match detector_type {
        DetectorType::YOLOV8 => (640, 640),
        DetectorType::RTDETR => (640, 640), // RT-DETR now uses 640×640 (same as YOLO)
    };

    let detector_config = DetectorConfig {
        fp16_model_path: Some(fp16_path.to_string()),
        fp32_model_path: Some(fp32_path.to_string()),
        confidence_threshold,
        nms_threshold: 0.45,
        input_size: (input_width, input_height),
        use_gpu: true,
        ..Default::default()
    };

    let frame_executor = Arc::new(FrameExecutor::new(
        detector_type,
        detector_config,
        executor_config,
    )?);
    println!("✓ ({:.2}s)", start_load.elapsed().as_secs_f32());

    let (tile_width, tile_height) = frame_executor.input_size();

    // Calculate expected tiles with detector-specific overlap
    let overlap = match detector_type {
        DetectorType::YOLOV8 => 32,
        DetectorType::RTDETR => 32, // RT-DETR uses smaller overlap for faster processing
    };
    let stride = tile_width - overlap;

    // Use same logic as pipeline: don't create extra tiles for small overhangs (< 20% of tile size)
    let tiles_x = if img_width <= tile_width {
        1
    } else {
        let remaining = img_width - tile_width;
        let num_strides = (remaining as f32 / stride as f32).ceil() as usize;
        let last_overhang = (remaining as i32) - ((num_strides - 1) * stride as usize) as i32;
        if last_overhang < (tile_width / 5) as i32 {
            num_strides
        } else {
            num_strides + 1
        }
    };

    let tiles_y = if img_height <= tile_height {
        1
    } else {
        let remaining = img_height - tile_height;
        let num_strides = (remaining as f32 / stride as f32).ceil() as usize;
        let last_overhang = (remaining as i32) - ((num_strides - 1) * stride as usize) as i32;
        if last_overhang < (tile_height / 5) as i32 {
            num_strides
        } else {
            num_strides + 1
        }
    };

    let total_tiles = tiles_x * tiles_y;

    println!("\n💡 Pipeline Configuration:");
    println!("   • Detector: {}", detector_name);
    println!(
        "   • Tile size: {}x{} (auto-detected from model)",
        tile_width, tile_height
    );
    println!(
        "   • Expected tiles: {} ({}×{}) with {}px overlap",
        total_tiles, tiles_x, tiles_y, overlap
    );
    println!(
        "   • Queue depth: {} (backpressure control)",
        max_queue_depth
    );
    println!(
        "   • Confidence threshold: {:.0}%",
        confidence_threshold * 100.0
    );
    println!(
        "   • Architecture: {} tiles → FrameExecutor → GPU inference",
        total_tiles
    );
    println!("   • Frame executor is shared and can process multiple images\n");

    // Create pipeline with shared frame executor
    println!("🔧 Building detection pipeline:");
    println!("  • Stage 1: Preprocess (tile extraction + shadow removal)");
    println!("  • Stage 2: Execution (frame detection via frame executor)");
    println!("  • Stage 3: Postprocess (NMS + nested filtering + coordinate merging)");

    let pipeline_config = PipelineConfig {
        overlap,
        allowed_classes: vec![0, 2, 4, 5, 7], // person, car, airplane, bus, truck
        iou_threshold: 0.5,
    };

    let pipeline = DetectionPipeline::new(Arc::clone(&frame_executor), pipeline_config);
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

    // Draw results on resized image (annotations match computation space)
    let mut annotated_img = result.resized_image.clone();
    print!(
        "🎨 Drawing bounding boxes on {}x{} image... ",
        annotated_img.width(),
        annotated_img.height()
    );

    for (detection_num, det) in result.detections.iter().enumerate() {
        let color = generate_class_color(det.class_id);

        // Use detection coordinates directly (already in resized image space)
        draw_rect(
            &mut annotated_img,
            det.x as i32,
            det.y as i32,
            det.w as u32,
            det.h as u32,
            color,
            2,
        );

        let label = format!(
            "{}#{} {:.0}%",
            det.class_name,
            detection_num + 1,
            det.confidence * 100.0
        );
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

    println!("\n💡 The batch executor can now be reused for more images!");
    println!("   Example: pipeline.process(&another_image)?;");

    Ok(())
}
