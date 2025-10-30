//! Main military target detector implementation

use crate::error::{DetectionError, Result};
use crate::postprocessing::Postprocessor;
use crate::preprocessing::ImagePreprocessor;
use crate::types::{DetectionResult, DetectorConfig, ImageData};

use ndarray::{s, Array4, ArrayView4};
use onnxruntime::{
    environment::Environment, session::Session, GraphOptimizationLevel, LoggingLevel,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Military target detector using ONNX Runtime
pub struct MilitaryTargetDetector<'a> {
    /// ONNX Runtime session
    session: Session<'a>,
    /// Image preprocessor
    preprocessor: ImagePreprocessor,
    /// Output postprocessor
    postprocessor: Postprocessor,
    /// Detector configuration
    config: DetectorConfig,
    /// Input tensor name
    input_name: String,
    /// Output tensor names
    output_names: Vec<String>,
}

impl<'a> MilitaryTargetDetector<'a> {
    /// Create new detector from configuration
    pub fn new(config: DetectorConfig) -> Result<Self> {
        log::info!("Initializing military target detector");
        log::info!("Model path: {}", config.model_path);
        log::info!("Input size: {:?}", config.input_size);
        log::info!("GPU enabled: {}", config.use_gpu);

        // Validate model file exists
        if !Path::new(&config.model_path).exists() {
            return Err(DetectionError::model_load(format!(
                "Model file not found: {}",
                config.model_path
            )));
        }

        // Create ONNX Runtime environment
        let environment = Arc::new(
            Environment::builder()
                .with_name("military_target_detector")
                .with_log_level(LoggingLevel::Warning)
                .build()?,
        );

        // Load model - use the basic constructor available in this version
        let model_data = std::fs::read(&config.model_path)
            .map_err(|e| DetectionError::model_load(format!("Failed to read model file: {}", e)))?;
        let session = Session::new(environment.clone(), model_data)?;

        // Get input/output tensor information
        let input_info = session
            .inputs
            .get(0)
            .ok_or_else(|| DetectionError::model_load("No input tensor found".to_string()))?;

        let input_name = input_info.name.clone();

        let output_names: Vec<String> = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        if output_names.is_empty() {
            return Err(DetectionError::model_load(
                "No output tensors found".to_string(),
            ));
        }

        log::info!("Input tensor: {}", input_name);
        log::info!("Output tensors: {:?}", output_names);

        // Validate input dimensions
        if let Some(shape) = &input_info.dimensions {
            if shape.len() != 4 {
                return Err(DetectionError::model_load(format!(
                    "Expected 4D input tensor, got {}D",
                    shape.len()
                )));
            }

            // Check if input size matches configuration
            let model_height = shape[2] as u32;
            let model_width = shape[3] as u32;

            if (model_width, model_height) != config.input_size {
                log::warn!(
                    "Model input size ({}, {}) differs from config ({}, {})",
                    model_width,
                    model_height,
                    config.input_size.0,
                    config.input_size.1
                );
            }
        }

        // Create preprocessor and postprocessor
        let preprocessor = ImagePreprocessor::new(config.input_size);

        let postprocessor = Postprocessor::new(
            config.confidence_threshold,
            config.nms_threshold,
            config.max_detections,
            config.input_size,
        );

        log::info!("Military target detector initialized successfully");

        Ok(Self {
            session,
            preprocessor,
            postprocessor,
            config,
            input_name,
            output_names,
        })
    }

    /// Create detector with default configuration
    pub fn default() -> Result<Self> {
        Self::new(DetectorConfig::default())
    }

    /// Load detector from model file with default settings
    pub fn from_model_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let mut config = DetectorConfig::default();
        config.model_path = model_path.as_ref().to_string_lossy().to_string();
        Self::new(config)
    }

    /// Detect targets in a single image
    pub fn detect(&self, image: &ImageData) -> Result<DetectionResult> {
        let start_time = Instant::now();

        // Preprocess image
        let input_tensor = self.preprocessor.preprocess(image)?;
        let original_size = (image.width, image.height);

        // Run inference
        let output = self.run_inference(&input_tensor)?;

        let inference_time = start_time.elapsed().as_secs_f32() * 1000.0; // Convert to ms

        // Postprocess results
        self.postprocessor
            .process(&output, original_size, inference_time)
    }

    /// Detect targets in multiple images (batch processing)
    pub fn detect_batch(&self, images: &[ImageData]) -> Result<Vec<DetectionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();

        // Preprocess all images
        let input_tensor = self.preprocessor.preprocess_batch(images)?;

        // Run batch inference
        let outputs = self.run_batch_inference(&input_tensor, images.len())?;

        let total_time = start_time.elapsed().as_secs_f32() * 1000.0; // Convert to ms
        let per_image_time = total_time / images.len() as f32;

        // Postprocess each result
        let mut results = Vec::new();
        for (i, image) in images.iter().enumerate() {
            let output = outputs.slice(s![i, .., ..]);
            let original_size = (image.width, image.height);

            let result =
                self.postprocessor
                    .process(&output.to_owned(), original_size, per_image_time)?;

            results.push(result);
        }

        Ok(results)
    }

    /// Load and detect targets from image file
    pub fn detect_file<P: AsRef<Path>>(&self, image_path: P) -> Result<DetectionResult> {
        let image = self.preprocessor.load_image(image_path)?;
        self.detect(&image)
    }

    /// Update detector configuration
    pub fn update_config(&mut self, config: DetectorConfig) -> Result<()> {
        // If model path changed, reload the model
        if config.model_path != self.config.model_path {
            *self = Self::new(config)?;
            return Ok(());
        }

        // Update postprocessor settings
        self.postprocessor.update_config(
            Some(config.confidence_threshold),
            Some(config.nms_threshold),
            Some(config.max_detections),
        );

        // Update config
        self.config = config;

        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }

    /// Warm up the model (run dummy inference)
    pub fn warmup(&self) -> Result<()> {
        log::info!("Warming up model...");

        // Create dummy image
        let dummy_data =
            vec![0u8; (self.config.input_size.0 * self.config.input_size.1 * 3) as usize];
        let dummy_image = ImageData::new(
            dummy_data,
            self.config.input_size.0,
            self.config.input_size.1,
            crate::types::ImageFormat::RGB,
        );

        // Run inference a few times
        for i in 0..3 {
            let start = Instant::now();
            self.detect(&dummy_image)?;
            let elapsed = start.elapsed().as_millis();
            log::debug!("Warmup iteration {}: {}ms", i + 1, elapsed);
        }

        log::info!("Model warmup completed");
        Ok(())
    }

    /// Run inference on preprocessed tensor
    fn run_inference(&self, input: &Array4<f32>) -> Result<ndarray::Array2<f32>> {
        use onnxruntime::tensor::OrtOwnedTensor;

        // Convert to ONNX tensor format - create owned tensor from input data
        let input_tensor = input.to_owned();

        // Run inference
        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(vec![input_tensor.into()])?;

        if outputs.is_empty() {
            return Err(DetectionError::inference(
                "No outputs from model".to_string(),
            ));
        }

        // Convert output to ndarray
        let output_tensor = &outputs[0];
        let output_view = output_tensor.view();

        // Expect 2D output: [num_detections, detection_data]
        if output_view.ndim() != 2 {
            return Err(DetectionError::inference(format!(
                "Expected 2D output, got {}D",
                output_view.ndim()
            )));
        }

        // Convert dynamic array to fixed 2D array
        let shape = output_view.shape();
        let owned = output_view.to_owned();
        let reshaped = owned
            .into_shape((shape[0], shape[1]))
            .map_err(|e| DetectionError::inference(format!("Failed to reshape output: {}", e)))?;

        Ok(reshaped)
    }

    /// Run batch inference
    fn run_batch_inference(
        &self,
        input: &Array4<f32>,
        batch_size: usize,
    ) -> Result<ndarray::Array3<f32>> {
        use onnxruntime::tensor::OrtOwnedTensor;

        // Convert to ONNX tensor format - create owned tensor from input data
        let input_tensor = input.to_owned();

        // Run inference
        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(vec![input_tensor.into()])?;

        if outputs.is_empty() {
            return Err(DetectionError::inference(
                "No outputs from model".to_string(),
            ));
        }

        // Convert output to ndarray
        let output_tensor = &outputs[0];
        let output_view = output_tensor.view();

        // Expect 3D output: [batch_size, num_detections, detection_data]
        if output_view.ndim() != 3 {
            return Err(DetectionError::inference(format!(
                "Expected 3D batch output, got {}D",
                output_view.ndim()
            )));
        }

        // Convert dynamic array to fixed 3D array
        let shape = output_view.shape();
        let owned = output_view.to_owned();
        let reshaped = owned
            .into_shape((shape[0], shape[1], shape[2]))
            .map_err(|e| {
                DetectionError::inference(format!("Failed to reshape batch output: {}", e))
            })?;

        Ok(reshaped)
    }

    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            input_name: self.input_name.clone(),
            input_size: self.config.input_size,
            output_names: self.output_names.clone(),
            confidence_threshold: self.config.confidence_threshold,
            nms_threshold: self.config.nms_threshold,
            max_detections: self.config.max_detections,
            use_gpu: self.config.use_gpu,
        }
    }
}

/// Information about the loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub input_name: String,
    pub input_size: (u32, u32),
    pub output_names: Vec<String>,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub max_detections: usize,
    pub use_gpu: bool,
}

// Re-export ndarray for convenience when using the library
pub use ndarray;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ImageFormat;

    // Note: These tests require a model file to be present
    // In practice, you would use a test model or mock the ONNX session

    #[test]
    #[ignore] // Ignore by default since it requires a model file
    fn test_detector_creation() {
        let config = DetectorConfig {
            model_path: "test_model.onnx".to_string(),
            ..Default::default()
        };

        // This would fail without an actual model file
        match MilitaryTargetDetector::new(config) {
            Ok(_detector) => {
                // Detector created successfully
            }
            Err(DetectionError::ModelLoadError(_)) => {
                // Expected when model file doesn't exist
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_config_validation() {
        let config = DetectorConfig::default();

        assert_eq!(config.input_size, (640, 640));
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.nms_threshold, 0.45);
        assert_eq!(config.max_detections, 100);
    }

    #[test]
    fn test_image_data_creation() {
        let data = vec![255u8; 640 * 640 * 3];
        let image = ImageData::new(data, 640, 640, ImageFormat::RGB);

        assert_eq!(image.width, 640);
        assert_eq!(image.height, 640);
        assert_eq!(image.channels(), 3);
        assert!(image.validate());
    }
}
