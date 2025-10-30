// Test the stub implementation functionality
use military_target_detector::{DetectorConfig, ImageData, ImageFormat, MilitaryTargetDetector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("🦀 Testing Rust Military Target Detector Library");
    println!("================================================");

    // Create detector configuration
    let config = DetectorConfig {
        model_path: "test_model.onnx".to_string(),
        input_size: (640, 640),
        confidence_threshold: 0.5,
        nms_threshold: 0.45,
        max_detections: 100,
        use_gpu: false, // Use CPU for testing
        gpu_device_id: 0,
        num_threads: Some(4),
        optimize_for_speed: true,
    };

    println!("✅ Configuration created");
    println!("   Model: {}", config.model_path);
    println!("   Input size: {:?}", config.input_size);
    println!("   GPU enabled: {}", config.use_gpu);

    // Create detector
    let detector = MilitaryTargetDetector::new(config)?;
    println!("✅ Detector created successfully");

    // Get model info
    let model_info = detector.model_info();
    println!("✅ Model info:");
    println!("   Input: {}", model_info.input_name);
    println!("   Input shape: {:?}", model_info.input_shape);
    println!("   Outputs: {:?}", model_info.output_names);

    // Test warmup
    detector.warmup()?;
    println!("✅ Model warmup completed");

    // Create dummy image data
    let image_data = ImageData::new(
        vec![0u8; 640 * 640 * 3], // Dummy RGB data
        640,
        640,
        ImageFormat::RGB,
    );
    println!(
        "✅ Test image created ({}x{} RGB)",
        image_data.width, image_data.height
    );

    // Test detection
    let result = detector.detect(&image_data)?;
    println!("✅ Detection completed");
    println!("   Detections found: {}", result.count());
    println!("   Inference time: {:.2}ms", result.inference_time_ms);
    println!(
        "   Image size: {}x{}",
        result.image_width, result.image_height
    );

    // Test batch detection
    let images = vec![&image_data];
    let batch_results = detector.detect_batch(
        &images
            .iter()
            .map(|img| img.data.as_slice())
            .collect::<Vec<_>>(),
    )?;
    println!("✅ Batch detection completed");
    println!("   Batch size: {}", batch_results.len());

    println!();
    println!("🎯 All tests passed!");
    println!(
        "📦 Library version: {}",
        military_target_detector::version()
    );
    println!(
        "🎖️  Target classes: {:?}",
        military_target_detector::get_target_classes()
            .iter()
            .map(|c| c.name())
            .collect::<Vec<_>>()
    );

    Ok(())
}
