//! GPU-accelerated military target detector using ONNX Runtime
//!
//! This implementation uses the official ONNX Runtime with CoreML/Metal backend
//! for GPU acceleration on macOS. Provides high-performance inference.

use crate::error::DetectionError;
use crate::types::{BoundingBox, Detection, DetectorConfig, ImageData, TargetClass};
#[cfg(feature = "metal")]
use log::warn;
use log::{debug, info};
use ndarray::{Array, ArrayView, IxDyn};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

/// GPU-accelerated military target detector
pub struct MilitaryTargetDetector {
    session: Session,
    config: DetectorConfig,
}

impl MilitaryTargetDetector {
    /// Create a new detector with GPU acceleration
    pub fn new(config: DetectorConfig) -> Result<Self, DetectionError> {
        info!("Initializing ONNX Runtime detector");
        info!("Model: {}", config.model_path);

        // Configure session with GPU support (v2.0 API)
        let builder = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?;

        // Try to use CoreML (Metal backend) on Apple platforms when metal feature is enabled
        #[cfg(feature = "metal")]
        let builder = {
            use ort::execution_providers::CoreMLExecutionProvider;
            info!("Attempting to use CoreML (Metal) backend...");

            builder
                .with_execution_providers([CoreMLExecutionProvider::default().build()])
                .unwrap_or_else(|e| {
                    warn!("CoreML not available: {}, using CPU", e);
                    Session::builder()
                        .expect("Failed to create fallback session builder")
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .expect("Failed to set optimization level")
                })
        };

        // Load the model
        let session = builder
            .commit_from_file(&config.model_path)
            .map_err(|e| DetectionError::ModelLoadError(format!("Failed to load model: {}", e)))?;

        info!("✓ Model loaded successfully");

        Ok(Self { session, config })
    }

    /// Detect targets in an image
    pub fn detect(&mut self, image: &ImageData) -> Result<Vec<Detection>, DetectionError> {
        debug!(
            "Starting detection on {}x{} image",
            image.width, image.height
        );

        // Preprocess: resize to 640x640, normalize, CHW format
        let input_tensor = self.preprocess(image)?;

        // Run inference using ort v2.0 API
        let tensor_ref = TensorRef::from_array_view(&input_tensor)
            .map_err(|e| DetectionError::InferenceError(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| DetectionError::InferenceError(e.to_string()))?;

        // Extract and own the output tensor
        let output_array = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| DetectionError::postprocessing(e.to_string()))?
            .into_owned();

        // Drop outputs to release borrow on self
        drop(outputs);

        // Postprocess: parse YOLOv8 output and apply NMS
        let detections = self.postprocess(output_array.view(), image.width, image.height)?;

        info!("Detected {} targets", detections.len());
        Ok(detections)
    }

    /// Preprocess image: resize to 640x640, normalize, convert to CHW format
    fn preprocess(&self, image: &ImageData) -> Result<Array<f32, IxDyn>, DetectionError> {
        use image::imageops::FilterType;
        use image::RgbImage;

        // Create RgbImage from raw bytes
        let rgb_img = RgbImage::from_raw(image.width, image.height, image.data.clone())
            .ok_or_else(|| {
                DetectionError::preprocessing("Failed to create image from raw data".to_string())
            })?;

        // Convert to DynamicImage for resizing
        let img = image::DynamicImage::ImageRgb8(rgb_img);

        // Resize to 640x640
        let resized = img.resize_exact(640, 640, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // Convert to normalized CHW format [1, 3, 640, 640]
        let mut input_data = Array::zeros((1, 3, 640, 640));

        for y in 0..640 {
            for x in 0..640 {
                let pixel = rgb.get_pixel(x, y);
                input_data[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                input_data[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                input_data[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }

        Ok(input_data.into_dyn())
    }

    /// Postprocess YOLOv8 output: parse detections and apply NMS
    fn postprocess(
        &self,
        output: ArrayView<f32, IxDyn>,
        orig_w: u32,
        orig_h: u32,
    ) -> Result<Vec<Detection>, DetectionError> {
        // YOLOv8 output shape can be:
        // [1, 5, 8400] for single-class models: [x, y, w, h, confidence]
        // [1, 7, 8400] for 3-class models: [x, y, w, h, conf, class1, class2, class3]
        // [1, 8+, 8400] for multi-class models

        let shape = output.shape();
        if shape.len() != 3 || shape[1] < 5 {
            return Err(DetectionError::postprocessing(format!(
                "Unexpected output shape: {:?}",
                shape
            )));
        }

        let num_channels = shape[1];
        let num_boxes = shape[2];
        let is_single_class = num_channels == 5;

        let mut detections = Vec::new();

        for i in 0..num_boxes {
            // Extract bbox coordinates
            let x_center = output[[0, 0, i]];
            let y_center = output[[0, 1, i]];
            let width = output[[0, 2, i]];
            let height = output[[0, 3, i]];

            // Handle confidence and class based on model type
            let (confidence, class_id) = if is_single_class {
                // Single-class model: confidence is at index 4
                let conf = output[[0, 4, i]];
                (conf, 0) // Default to class 0
            } else {
                // Multi-class model: find best class (channels after bbox)
                let num_classes = num_channels - 4;
                let mut max_score = 0.0f32;
                let mut max_class = 0;
                for c in 0..num_classes {
                    let score = output[[0, 4 + c, i]];
                    if score > max_score {
                        max_score = score;
                        max_class = c;
                    }
                }
                (max_score, max_class)
            };

            // Filter by confidence threshold
            if confidence < self.config.confidence_threshold {
                continue;
            }

            // Convert to TargetClass
            let class = TargetClass::from_id(class_id as u32).ok_or_else(|| {
                DetectionError::postprocessing(format!("Invalid class ID: {}", class_id))
            })?;

            // Convert to normalized coordinates [0, 1]
            let scale_x = orig_w as f32 / 640.0;
            let scale_y = orig_h as f32 / 640.0;

            let x = ((x_center - width / 2.0) * scale_x / orig_w as f32)
                .max(0.0)
                .min(1.0);
            let y = ((y_center - height / 2.0) * scale_y / orig_h as f32)
                .max(0.0)
                .min(1.0);
            let w = ((width * scale_x) / orig_w as f32).max(0.0).min(1.0 - x);
            let h = ((height * scale_y) / orig_h as f32).max(0.0).min(1.0 - y);

            let bbox = BoundingBox::new(x, y, w, h);
            detections.push(Detection::new(class, confidence, bbox));
        }

        // Apply Non-Maximum Suppression
        let filtered = self.non_max_suppression(detections);

        Ok(filtered)
    }

    /// Non-Maximum Suppression to remove overlapping detections
    fn non_max_suppression(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();

        while !detections.is_empty() {
            let current = detections.remove(0);
            keep.push(current.clone());

            detections.retain(|det| {
                // Keep if different class or low IoU
                if det.class != current.class {
                    return true;
                }

                let iou = current.bbox.iou(&det.bbox);
                iou < self.config.nms_threshold
            });
        }

        keep
    }
}
