// Temporary stub implementation for ONNX Runtime compatibility
// This file provides minimal stubs to make the project compile
// TODO: Implement proper ONNX Runtime integration once API compatibility is resolved

use crate::error::Result;
use crate::types::DetectorConfig;
use ndarray::{Array2, Array3, Array4};

/// Stub implementation of military target detector
pub struct MilitaryTargetDetector {
    config: DetectorConfig,
}

impl MilitaryTargetDetector {
    /// Create new detector from configuration
    pub fn new(config: DetectorConfig) -> Result<Self> {
        log::info!("Creating stub detector (ONNX Runtime integration pending)");
        Ok(Self { config })
    }

    /// Detect targets in single image
    pub fn detect_single(&self, _image_data: &[u8]) -> Result<Vec<crate::types::Detection>> {
        log::warn!("Stub implementation - no actual detection performed");
        Ok(Vec::new())
    }

    /// Detect targets in batch of images
    pub fn detect_batch(&self, _images: &[&[u8]]) -> Result<Vec<Vec<crate::types::Detection>>> {
        log::warn!("Stub implementation - no batch detection performed");
        Ok(Vec::new())
    }

    /// Warm up the model
    pub fn warmup(&self) -> Result<()> {
        log::info!("Stub warmup completed");
        Ok(())
    }

    /// Get model information
    pub fn model_info(&self) -> crate::types::ModelInfo {
        crate::types::ModelInfo {
            input_name: "stub_input".to_string(),
            output_names: vec!["stub_output".to_string()],
            input_shape: vec![1, 3, 640, 640],
            output_shapes: vec![vec![1, 25200, 85]],
            model_path: self.config.model_path.clone(),
        }
    }

    /// Run inference stub
    #[allow(dead_code)]
    fn run_inference(&mut self, _input: &Array4<f32>) -> Result<Array2<f32>> {
        // Return dummy output for now
        Ok(Array2::zeros((1, 85)))
    }

    /// Run batch inference stub  
    #[allow(dead_code)]
    fn run_batch_inference(
        &mut self,
        _input: &Array4<f32>,
        _batch_size: usize,
    ) -> Result<Array3<f32>> {
        // Return dummy output for now
        Ok(Array3::zeros((1, 1, 85)))
    }

    /// Detect targets in image (unified method for FFI)
    pub fn detect(
        &self,
        _image: &crate::types::ImageData,
    ) -> Result<crate::types::DetectionResult> {
        log::warn!("Stub detect method called");
        Ok(crate::types::DetectionResult::new(
            Vec::new(),
            0.0,
            640,
            640,
        ))
    }

    /// Detect targets from file path
    pub fn detect_file(&self, _path: &str) -> Result<crate::types::DetectionResult> {
        log::warn!("Stub detect_file method called");
        Ok(crate::types::DetectionResult::new(
            Vec::new(),
            0.0,
            640,
            640,
        ))
    }

    /// Get detector configuration
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }

    /// Update detector configuration
    pub fn update_config(&mut self, config: DetectorConfig) -> Result<()> {
        log::info!("Updating stub detector configuration");
        self.config = config;
        Ok(())
    }
}
