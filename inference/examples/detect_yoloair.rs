use image::Rgb;
use military_target_detector::image_utils::{
    draw_rect, draw_text, generate_class_color, resize_with_letterbox,
};
use military_target_detector::types::{DetectorConfig, ImageData};
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

/// Apply simple HSV-based shadow removal (brightness enhancement)
/// Increases V channel for darker pixels (shadows) more than bright pixels
/// Input: RgbImage
fn remove_shadows_hsv(img: &image::RgbImage) -> image::RgbImage {
    let (width, height) = img.dimensions();
    let mut result = image::RgbImage::new(width, height);

    // Convert RGB to HSV, enhance V channel, convert back
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let [r, g, b] = pixel.0;

            // RGB to HSV conversion
            let r_f = r as f32 / 255.0;
            let g_f = g as f32 / 255.0;
            let b_f = b as f32 / 255.0;

            let max = r_f.max(g_f).max(b_f);
            let min = r_f.min(g_f).min(b_f);
            let delta = max - min;

            // Calculate H (hue)
            let h = if delta < 0.00001 {
                0.0
            } else if (max - r_f).abs() < 0.00001 {
                60.0 * (((g_f - b_f) / delta) % 6.0)
            } else if (max - g_f).abs() < 0.00001 {
                60.0 * (((b_f - r_f) / delta) + 2.0)
            } else {
                60.0 * (((r_f - g_f) / delta) + 4.0)
            };

            // Calculate S (saturation)
            let s = if max < 0.00001 { 0.0 } else { delta / max };

            // V (value/brightness)
            let v = max;

            // Enhance V channel for dark pixels (shadows)
            // Increase brightness more for darker pixels
            let v_enhanced = if v < 0.5 {
                // Dark pixels: increase brightness significantly
                (v + 0.3).min(1.0)
            } else {
                // Bright pixels: slight increase
                (v + 0.1).min(1.0)
            };

            // HSV back to RGB
            let c = v_enhanced * s;
            let x_val = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
            let m = v_enhanced - c;

            let (r_out, g_out, b_out) = if h < 60.0 {
                (c, x_val, 0.0)
            } else if h < 120.0 {
                (x_val, c, 0.0)
            } else if h < 180.0 {
                (0.0, c, x_val)
            } else if h < 240.0 {
                (0.0, x_val, c)
            } else if h < 300.0 {
                (x_val, 0.0, c)
            } else {
                (c, 0.0, x_val)
            };

            let r_final = ((r_out + m) * 255.0).clamp(0.0, 255.0) as u8;
            let g_final = ((g_out + m) * 255.0).clamp(0.0, 255.0) as u8;
            let b_final = ((b_out + m) * 255.0).clamp(0.0, 255.0) as u8;

            result.put_pixel(x, y, Rgb([r_final, g_final, b_final]));
        }
    }

    result
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
        confidence_threshold: 0.30, // EXTREMELY low threshold (22%) to catch small objects
        nms_threshold: 0.45,
        input_size: (512, 512), // Match shadow removal output size
        use_gpu: true,
        ..Default::default()
    };

    println!("⚙️  Model: yolov8m_world_detector.onnx");
    println!("⚙️  Confidence threshold: 22% (extreme diagnostic mode)");

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

    // Preprocessing Stage 2: Apply HSV-based shadow removal (brightness enhancement)
    print!("🌓 Preprocessing: Applying HSV brightness enhancement... ");
    let start_shadow = Instant::now();
    let preprocessed_img = remove_shadows_hsv(&stage1_img);
    let elapsed = start_shadow.elapsed().as_secs_f32() * 1000.0;
    println!("✓ ({:.0}ms)", elapsed);

    // Save shadow-removed image for inspection at fixed path
    let shadow_output_path = "output_shadowremoved.jpg";
    if let Err(e) = preprocessed_img.save(shadow_output_path) {
        println!("⚠️  Could not save enhanced image: {}", e);
    } else {
        println!("💾 Saved enhanced image: {}", shadow_output_path);
    }

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
