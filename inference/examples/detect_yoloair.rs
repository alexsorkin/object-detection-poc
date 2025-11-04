use image::Rgb;
use image::RgbImage;
use military_target_detector::image_utils::{
    draw_rect, draw_text, generate_class_color, resize_with_letterbox,
};
use military_target_detector::types::{DetectorConfig, ImageData};
use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Value;
use std::env;
use std::time::Instant;

/// Single image detection example with ONNX Runtime + GPU (YOLO Aerial Detector)
///
/// This example demonstrates:
/// - Loading and running inference on aerial images
/// - Drawing bounding boxes on detected objects
/// - Saving annotated output image
///
/// Usage:
///   cargo run --release --features metal --example detect_yoloair [image_path] [output_path]
///   
/// Example:
///   cargo run --release --features metal --example detect_yoloair test.jpg output.jpg

/// Detect shadow regions in an image and return a binary mask
/// Shadows are typically: dark (low V), low saturation, and not strongly colored
fn detect_shadow_mask(img: &RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut mask = RgbImage::new(width, height);

    // Parameters for shadow detection (relaxed to detect more shadows)
    let shadow_value_max = 0.50; // V < 0.50 (darker pixels - increased from 0.35)
    let shadow_saturation_max = 0.35; // S < 0.35 (low saturation - increased from 0.25)
    let shadow_brightness_max = 100.0; // Brightness < 100 (darker - increased from 90)

    // Additional: check color consistency (shadows should be fairly uniform gray)
    let color_variance_threshold = 35.0; // Low variance in RGB values (relaxed from 25)

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Convert RGB to HSV
            let max_rgb = r.max(g).max(b);
            let min_rgb = r.min(g).min(b);
            let delta = max_rgb - min_rgb;

            // Value (brightness)
            let v = max_rgb;

            // Saturation
            let s = if max_rgb > 0.0 { delta / max_rgb } else { 0.0 };

            // Brightness (perceived luminance)
            let brightness =
                0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;

            // Color variance (difference between max and min RGB values)
            let color_variance = (pixel[0] as f32 - pixel[1] as f32).abs()
                + (pixel[1] as f32 - pixel[2] as f32).abs()
                + (pixel[0] as f32 - pixel[2] as f32).abs();

            // Detect shadow: dark AND low saturation AND low brightness AND low color variance
            // This targets actual shadow regions, not dirt/stains which have more color variation
            let is_shadow = v < shadow_value_max
                && s < shadow_saturation_max
                && brightness < shadow_brightness_max
                && color_variance < color_variance_threshold;

            // Mask: white (255) for shadow regions, black (0) for non-shadow
            let mask_value = if is_shadow { 255 } else { 0 };
            mask.put_pixel(x, y, Rgb([mask_value, mask_value, mask_value]));
        }
    }

    mask
}

/// Apply CNN-based shadow removal using ONNX model
/// Input: Already resized/letterboxed image
fn remove_shadows_cnn(img: &RgbImage) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let model_size = img.width(); // Use input image size (should be 512x512)

    // Step 1: Detect shadow mask on the input image
    println!(
        "  Detecting shadows on {}x{} image...",
        model_size, model_size
    );
    let shadow_mask = detect_shadow_mask(img);

    // Save shadow mask for inspection
    if let Err(e) = shadow_mask.save("output_shadow_mask.jpg") {
        println!("  ⚠️  Could not save shadow mask: {}", e);
    } else {
        println!("  💾 Saved shadow mask: output_shadow_mask.jpg");
    }

    // Load shadow removal model (5.6MB FP16 model with full architecture)
    use ort::execution_providers::CoreMLExecutionProvider;

    let mut session = Session::builder()?
        .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        .commit_from_file("../models/shadowmaskformer_fp32.onnx")?;

    println!("  Model loaded successfully");
    println!(
        "  Input names: {:?}",
        session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
    );
    println!(
        "  Output names: {:?}",
        session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
    );

    // Prepare input: convert RGB image to normalized float tensor [1, 3, H, W]
    // IMPORTANT: Model requires [-1, 1] normalization, not [0, 1]!
    let mut input_data = Vec::with_capacity((3 * model_size * model_size) as usize);

    // Normalize to [-1, 1] and arrange as CHW format
    for c in 0..3 {
        for y in 0..model_size {
            for x in 0..model_size {
                let pixel = img.get_pixel(x, y);
                let value = (pixel[c as usize] as f32 / 255.0) * 2.0 - 1.0; // [0,1] -> [-1,1]
                input_data.push(value);
            }
        }
    }

    // Prepare mask from detected shadow regions (normalized to [-1, 1])
    // Mask: 1.0 for shadow regions (white in mask), -1.0 for non-shadow (black in mask)
    let mut mask_data = Vec::with_capacity((3 * model_size * model_size) as usize);

    for _c in 0..3 {
        // Same mask for all 3 channels
        for y in 0..model_size {
            for x in 0..model_size {
                let pixel = shadow_mask.get_pixel(x, y);
                // If shadow (white = 255), use 1.0, if non-shadow (black = 0), use -1.0
                let is_shadow = pixel[0] > 128;
                let value = if is_shadow { 1.0f32 } else { -1.0f32 };
                mask_data.push(value);
            }
        }
    }

    // Create input tensors [1, 3, 512, 512]
    let input_shape: Vec<usize> = vec![1, 3, model_size as usize, model_size as usize];
    let input_tensor = ArrayD::from_shape_vec(IxDyn(&input_shape), input_data)?;
    let mask_tensor = ArrayD::from_shape_vec(IxDyn(&input_shape), mask_data)?;

    // Run inference with TWO inputs: image and mask
    let outputs = session.run(ort::inputs![
        "input" => Value::from_array(input_tensor)?,
        "mask" => Value::from_array(mask_tensor)?
    ])?;

    // Extract output
    let output = outputs["output"].try_extract_array::<f32>()?;
    let output_data = output.view();

    println!("  Output shape: {:?}", output_data.shape());
    println!(
        "  Output range: [{:.3}, {:.3}]",
        output_data.iter().cloned().fold(f32::INFINITY, f32::min),
        output_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Convert back to RGB image at 512x512
    // IMPORTANT: Output is in [-1, 1] range, convert back to [0, 255]
    let mut result_512 = RgbImage::new(model_size, model_size);
    for y in 0..model_size {
        for x in 0..model_size {
            let r = ((output_data[[0, 0, y as usize, x as usize]] + 1.0) / 2.0 * 255.0)
                .clamp(0.0, 255.0) as u8;
            let g = ((output_data[[0, 1, y as usize, x as usize]] + 1.0) / 2.0 * 255.0)
                .clamp(0.0, 255.0) as u8;
            let b = ((output_data[[0, 2, y as usize, x as usize]] + 1.0) / 2.0 * 255.0)
                .clamp(0.0, 255.0) as u8;
            result_512.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    // Return 512x512 image directly (no resizing back)
    // The detector will use this size directly
    Ok(result_512)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let image_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "test_data/yolo_airport.jpg".to_string() // Use normalized 640x640 image
    };

    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        "output_annotated.jpg".to_string()
    };

    println!("🎯 YOLO Aerial Detector - GPU Accelerated\n");
    println!("⚙️  Backend: ONNX Runtime (CoreML/Metal on macOS)\n");

    // Configuration for aerial detection model
    // DIAGNOSTIC MODE: Extremely low threshold to detect even small/distant objects
    // Using 512x512 to match shadow removal output (no transformation)
    let config = DetectorConfig {
        model_path: "../models/yolov8m_world_detector.onnx".to_string(),
        confidence_threshold: 0.10, // EXTREMELY low threshold (10%) to catch small objects
        nms_threshold: 0.45,
        input_size: (512, 512), // Match shadow removal output size
        use_gpu: true,
        ..Default::default()
    };

    println!("⚙️  Model: yolov8m_world_detector.onnx");
    println!("⚙️  Confidence threshold: 10% (extreme diagnostic mode)");

    // Load original image (for drawing output)
    print!("📷 Loading image... ");
    let original_img = image::open(&image_path)?.to_rgb8();
    println!("✓ ({}x{})", original_img.width(), original_img.height());

    // Preprocessing Stage 1: Resize to 512x512 with letterboxing (for display and detection)
    print!("📐 Preprocessing: Resize to 512x512... ");
    let stage1_img = resize_with_letterbox(&original_img, 512);
    println!("✓ ({}x{})", stage1_img.width(), stage1_img.height());

    // Save stage 1 output for inspection
    if let Err(e) = stage1_img.save("output_stage1_resized.jpg") {
        println!("  ⚠️  Could not save stage1 image: {}", e);
    } else {
        println!("💾 Saved stage1 (resized): output_stage1_resized.jpg");
    }

    // Preprocessing Stage 2: Apply CNN-based shadow removal on 512x512 image
    print!("🌓 Preprocessing: Removing shadows... ");
    let start_shadow = Instant::now();
    let preprocessed_img = match remove_shadows_cnn(&stage1_img) {
        Ok(img) => {
            let elapsed = start_shadow.elapsed().as_secs_f32() * 1000.0;
            println!("✓ ({:.0}ms)", elapsed);

            // Save shadow-removed image for inspection at fixed path
            let shadow_output_path = "output_shadowremoved.jpg";
            if let Err(e) = img.save(shadow_output_path) {
                println!("⚠️  Could not save shadow-removed image: {}", e);
            } else {
                println!("💾 Saved shadow-removed image: {}", shadow_output_path);
            }

            img
        }
        Err(e) => {
            println!("⚠️  Failed: {}, using original image", e);
            original_img.clone()
        }
    };

    // Convert image to ImageData for detection
    use military_target_detector::types::ImageFormat;
    let image = ImageData {
        data: preprocessed_img.clone().into_raw(),
        width: preprocessed_img.width(),
        height: preprocessed_img.height(),
        format: ImageFormat::RGB,
    };

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

    // Filter detections to only include specific classes
    let allowed_classes = vec![
        "car",        // COCO class ID 2
        "motorcycle", // COCO class ID 3
        "airplane",   // COCO class ID 4
        "truck",      // COCO class ID 7
                      // Add more classes as needed:
                      // "bus",       // COCO class ID 5
                      // "person",    // COCO class ID 0
    ];

    let total_detections = detections.len();

    // First filter: Class filtering
    let detections: Vec<_> = detections
        .into_iter()
        .filter(|det| {
            let class_name = det.class.name();
            allowed_classes.contains(&class_name.as_str())
        })
        .collect();

    let class_filtered_count = total_detections - detections.len();

    // Shadow filtering is DISABLED because we're already doing CNN-based shadow removal preprocessing
    // Keeping this code commented for reference:
    // let detections: Vec<_> = detections
    //     .into_iter()
    //     .filter(|det| !is_likely_shadow(det, &stage1_img, image.width, image.height))
    //     .collect();

    let shadow_filtered_count = 0; // Disabled

    // Print results
    println!("\n📊 Results:");
    println!("  Total detections: {}", total_detections);
    if class_filtered_count > 0 {
        println!("  Filtered (other classes): {}", class_filtered_count);
    }
    if shadow_filtered_count > 0 {
        println!("  Filtered (likely shadows): {}", shadow_filtered_count);
    }
    println!("  Final detections: {}", detections.len());
    println!("  Allowed: {}", allowed_classes.join(", "));
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
        println!("    Class: {} (ID: {})", det.class.name(), det.class.id());
        println!("    Confidence: {:.1}%", det.confidence * 100.0);
        println!(
            "    Box: x={:.3}, y={:.3}, w={:.3}, h={:.3}",
            det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
        );
    }

    // Draw bounding boxes on Stage 1 preprocessed image (512x512 letterboxed)
    print!("\n🎨 Drawing bounding boxes on stage1 image (512x512)... \n");
    let mut output_img = stage1_img.clone();

    // Track serial numbers per class
    let mut class_counters: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for det in &detections {
        // Convert normalized coordinates to pixel coordinates
        let pixel_bbox = det.bbox.to_pixels(image.width, image.height);

        let class_id = det.class.id();

        // Increment counter for this class
        let serial_number = class_counters.entry(class_id).or_insert(0);
        *serial_number += 1;

        // Generate deterministic color based on class ID
        let color = generate_class_color(class_id);

        // Draw rectangle with thickness 3
        draw_rect(
            &mut output_img,
            pixel_bbox.x as i32,
            pixel_bbox.y as i32,
            pixel_bbox.width,
            pixel_bbox.height,
            color,
            1,
        );

        // Draw label with serial number instead of confidence percentage
        let label = format!("{} #{}", det.class.name(), serial_number);

        // Calculate label position
        let padding = 3;
        let text_x = (pixel_bbox.x as i32 + padding).max(0);
        let text_y = (pixel_bbox.y as i32 + padding).max(0);

        // Check if there's enough space for label
        if pixel_bbox.height > 20 && pixel_bbox.width > 30 {
            // Draw white text with transparent background
            let text_color = Rgb([255, 255, 255]); // White text
            draw_text(
                &mut output_img,
                &label,
                text_x,
                text_y,
                text_color,
                None, // Transparent background
            );
        }

        println!(
            "  ✓ Drew box for: {} #{} at ({}, {}) size {}x{}",
            det.class.name(),
            serial_number,
            pixel_bbox.x,
            pixel_bbox.y,
            pixel_bbox.width,
            pixel_bbox.height
        );
    }

    // Save annotated image
    output_img.save(&output_path)?;
    println!("\n✓ Drawing complete");
    println!("💾 Saved annotated image to: {}", output_path);

    Ok(())
}
