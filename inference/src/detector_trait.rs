/// Unified detector trait for different detection architectures
///
/// This trait allows the pipeline to work with any detector architecture
/// (RT-DETR, etc.) as long as they implement this interface.
use crate::types::{Detection, ImageData};

/// Common interface for object detectors
pub trait Detector: Send {
    /// Detect objects in a single image
    fn detect(&mut self, image: &ImageData) -> Result<Vec<Detection>, String>;

    /// Detect objects in multiple images (batch inference)
    fn detect_batch(&mut self, images: &[ImageData]) -> Result<Vec<Vec<Detection>>, String>;

    /// Get the detector name (for logging/debugging)
    fn name(&self) -> &str;

    /// Get the input size expected by the detector
    fn input_size(&self) -> (u32, u32);
}

/// Wrap RT-DETR detector to implement the Detector trait
impl Detector for crate::detector_rtdetr::RTDETRDetector {
    fn detect(&mut self, image: &ImageData) -> Result<Vec<Detection>, String> {
        self.detect(image).map_err(|e| e.to_string())
    }

    fn detect_batch(&mut self, images: &[ImageData]) -> Result<Vec<Vec<Detection>>, String> {
        self.detect_batch(images).map_err(|e| e.to_string())
    }

    fn name(&self) -> &str {
        "RT-DETR"
    }

    fn input_size(&self) -> (u32, u32) {
        (576, 576)
    }
}

/// Detector type enum for selecting which detector to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorType {
    RTDETR,
}

impl DetectorType {
    /// Create a detector instance of this type
    pub fn create(
        &self,
        config: crate::types::DetectorConfig,
    ) -> Result<Box<dyn Detector>, Box<dyn std::error::Error>> {
        match self {
            DetectorType::RTDETR => {
                let detector = crate::detector_rtdetr::RTDETRDetector::new(config)?;
                Ok(Box::new(detector))
            }
        }
    }
}
