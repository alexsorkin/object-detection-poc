// Test inference pipeline
use military_target_detector::{DetectorConfig, ImageData, ImageFormat, MilitaryTargetDetector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Military Target Detection - Inference Pipeline Test");
    println!("====================================================\n");

    // Test 1: Configuration
    println!("Test 1: Configuration Validation");
    let mut config = DetectorConfig::default();
    config.model_path = "models/military_targets.onnx".to_string();
    config.use_gpu = false;
    println!("[OK] Configuration created\n");

    // Test 2: Detector initialization
    println!("Test 2: Detector Initialization");
    let detector = MilitaryTargetDetector::new(config.clone())?;
    println!("[OK] Detector initialized\n");

    // Test 3: Target classes
    println!("Test 3: Target Classes");
    let classes = military_target_detector::get_target_classes();
    println!("[OK] {} target classes available\n", classes.len());

    // Test 4: Image data
    println!("Test 4: Image Data Creation");
    let image_data = ImageData::new(vec![128u8; 640 * 640 * 3], 640, 640, ImageFormat::RGB);
    println!("[OK] Test image created ({}x{})\n", image_data.width, image_data.height);

    // Test 5: Detection
    println!("Test 5: Single Image Detection");
    let result = detector.detect(&image_data)?;
    println!("[OK] Detection completed - {} detections found\n", result.count());

    // Test 6: Batch detection
    println!("Test 6: Batch Detection");
    let batch_images: Vec<_> = (0..4).map(|_| vec![50u8; 640 * 640 * 3]).collect();
    let batch_slices: Vec<_> = batch_images.iter().map(|img| img.as_slice()).collect();
    let batch_results = detector.detect_batch(&batch_slices)?;
    println!("[OK] Batch detection completed - {} images processed\n", batch_results.len());

    // Summary
    println!("======================");
    println!("All tests passed!");
    println!("Library version: {}", military_target_detector::version());
    println!("======================");

    Ok(())
}
