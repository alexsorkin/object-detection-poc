//! Military Target Detection Library
//!
//! A high-performance, cross-platform library for real-time military target detection
//! in video streams. Designed for integration with game engines, VR/AR applications,
//! and embedded systems.

// pub mod detector;  // Temporarily disabled due to ONNX Runtime API compatibility issues
pub mod detector_stub; // Temporary stub implementation
pub mod error;
pub mod ffi;
pub mod postprocessing;
pub mod preprocessing;
pub mod types;

#[cfg(feature = "opencv")]
pub mod video;

pub use detector_stub::MilitaryTargetDetector; // Use stub temporarily
pub use error::{DetectionError, Result};
pub use types::{
    BoundingBox, Detection, DetectionResult, DetectorConfig, ImageData, ImageFormat, ModelInfo,
    TargetClass,
};

/// Initialize the detection library
/// This function should be called once before using the detector
pub fn init() -> Result<()> {
    // ONNX Runtime initialization is now automatic in newer versions
    log::info!("Military target detection library initialized");
    Ok(())
}

/// Get library version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get supported target classes
pub fn get_target_classes() -> Vec<TargetClass> {
    vec![
        TargetClass::ArmedPersonnel,
        TargetClass::RocketLauncher,
        TargetClass::MilitaryVehicle,
        TargetClass::HeavyWeapon,
    ]
}
