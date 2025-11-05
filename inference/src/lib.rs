//! Military Target Detection Library
//!
//! A high-performance, cross-platform library for real-time military target detection
//! in video streams. Uses ONNX Runtime with GPU acceleration (CoreML/Metal on macOS).
//!
//! ## GPU Acceleration
//!
//! This library uses the official ONNX Runtime with:
//! - **macOS**: CoreML (Metal backend) for GPU acceleration
//! - **CUDA**: Nvidia GPUs (Linux/Windows)
//! - **CPU**: Fallback for all platforms
//!
//! ### Performance
//!
//! - **AMD Radeon Pro 5500M (CoreML)**: ~80-100ms inference
//! - **CPU fallback**: ~2,500ms inference
//!
//! ## Usage
//!
//! ```bash
//! cargo build --release
//! cargo run --release --example detect
//! ```

// Detector implementation
pub mod batch_executor;
pub mod detector_pool;
pub mod detector_rtdetr;
pub mod detector_trait;
pub mod detector_yolov8;
pub mod pipeline;

// Core modules
pub mod error;
pub mod ffi; // Unity/C# integration
pub mod image_utils; // Image manipulation utilities
pub mod postprocessing;
pub mod preprocessing;
pub mod types;

// Export the detectors
pub use detector_rtdetr::RTDETRDetector;
pub use detector_trait::{Detector, DetectorType};
pub use detector_yolov8::YoloV8Detector;

pub use error::{DetectionError, Result};
pub use types::{
    BoundingBox, Detection, DetectionResult, DetectorConfig, ImageData, ImageFormat, ModelInfo,
    TargetClass,
};

/// Initialize the detection library
pub fn init() -> Result<()> {
    log::info!("Military target detection library initialized (ONNX Runtime)");
    Ok(())
}

/// Get library version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get supported target classes
pub fn get_target_classes() -> Vec<TargetClass> {
    TargetClass::all()
}
