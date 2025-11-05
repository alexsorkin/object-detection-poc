//! GPU-accelerated YOLOv8 detector using ONNX Runtime
//!
//! This implementation uses the official ONNX Runtime with CoreML/Metal backend
//! for GPU acceleration on macOS. Provides high-performance inference.

use crate::error::DetectionError;
use crate::types::{BoundingBox, Detection, DetectorConfig, ImageData, TargetClass};
use log::{debug, info, warn};
use ndarray::{Array, ArrayView, IxDyn};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

/// GPU-accelerated YOLOv8 detector
pub struct YoloV8Detector {
    session: Session,
    config: DetectorConfig,
}

impl YoloV8Detector {
    /// Create a new detector with automatic FP16/FP32 model selection
    ///
    /// Selection logic:
    /// 1. If use_gpu=true and CoreML available: Use FP16 model with GPU
    /// 2. If use_gpu=true but CoreML fails: Fallback to FP32 model with CPU
    /// 3. If use_gpu=false: Use FP32 model with CPU
    pub fn new(config: DetectorConfig) -> Result<Self, DetectionError> {
        info!("Initializing ONNX Runtime detector");

        // Determine which model to use
        let (model_path, use_gpu_backend) = Self::select_model(&config)?;
        info!("Selected model: {}", model_path);

        // Try GPU path first if requested
        if config.use_gpu && use_gpu_backend {
            // Try CUDA first (NVIDIA GPUs)
            #[cfg(feature = "cuda")]
            {
                match Self::try_create_cuda_session(&model_path, &config) {
                    Ok(session) => {
                        info!("✓ Model loaded successfully with CUDA (NVIDIA GPU)");
                        return Ok(Self { session, config });
                    }
                    Err(e) => {
                        warn!("CUDA initialization failed: {}", e);
                        warn!("Trying TensorRT fallback...");
                    }
                }
            }

            // Try TensorRT (NVIDIA GPUs - optimized)
            #[cfg(feature = "tensorrt")]
            {
                match Self::try_create_tensorrt_session(&model_path, &config) {
                    Ok(session) => {
                        info!("✓ Model loaded successfully with TensorRT (NVIDIA GPU)");
                        return Ok(Self { session, config });
                    }
                    Err(e) => {
                        warn!("TensorRT initialization failed: {}", e);
                    }
                }
            }

            // Try CoreML (Apple Metal GPUs)
            #[cfg(feature = "metal")]
            {
                match Self::try_create_gpu_session(&model_path, &config) {
                    Ok(session) => {
                        info!("✓ Model loaded successfully with CoreML (GPU/Metal)");
                        return Ok(Self { session, config });
                    }
                    Err(e) => {
                        warn!("CoreML initialization failed: {}", e);
                        warn!("Falling back to CPU with FP32 model...");
                    }
                }
            }

            #[cfg(not(any(feature = "metal", feature = "cuda", feature = "tensorrt")))]
            {
                warn!("No GPU backend feature enabled (metal/cuda/tensorrt), falling back to CPU");
            }
        }

        // CPU fallback path
        let fallback_model = Self::get_fallback_model(&config)?;
        info!("Using CPU fallback model: {}", fallback_model);

        let session = Self::create_cpu_session(&fallback_model)?;
        info!("✓ Model loaded successfully with CPU");

        Ok(Self { session, config })
    }

    /// Select appropriate model based on config
    fn select_model(config: &DetectorConfig) -> Result<(String, bool), DetectionError> {
        // If new fields are set, use them
        if let Some(ref fp16_path) = config.fp16_model_path {
            if config.use_gpu {
                return Ok((fp16_path.clone(), true));
            }
        }

        if let Some(ref fp32_path) = config.fp32_model_path {
            if !config.use_gpu {
                return Ok((fp32_path.clone(), false));
            }
        }

        // Fallback to old model_path for backward compatibility
        #[allow(deprecated)]
        if !config.model_path.is_empty() {
            return Ok((config.model_path.clone(), config.use_gpu));
        }

        Err(DetectionError::ModelLoadError(
            "No model path specified. Set fp16_model_path/fp32_model_path or model_path"
                .to_string(),
        ))
    }

    /// Get fallback model path (FP32 for CPU)
    fn get_fallback_model(config: &DetectorConfig) -> Result<String, DetectionError> {
        if let Some(ref fp32_path) = config.fp32_model_path {
            return Ok(fp32_path.clone());
        }

        #[allow(deprecated)]
        if !config.model_path.is_empty() {
            return Ok(config.model_path.clone());
        }

        Err(DetectionError::ModelLoadError(
            "No fallback model path (fp32_model_path) specified".to_string(),
        ))
    }

    /// Try to create a GPU-accelerated session with CUDA (NVIDIA)
    #[cfg(feature = "cuda")]
    fn try_create_cuda_session(
        model_path: &str,
        config: &DetectorConfig,
    ) -> Result<Session, DetectionError> {
        use ort::execution_providers::CUDAExecutionProvider;

        info!("Attempting to use CUDA backend (NVIDIA GPU)...");

        let session = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(config.gpu_device_id)
                .build()])
            .map_err(|e| DetectionError::ModelLoadError(format!("CUDA provider failed: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| {
                DetectionError::ModelLoadError(format!("Failed to load model with CUDA: {}", e))
            })?;

        Ok(session)
    }

    /// Try to create a GPU-accelerated session with TensorRT (NVIDIA - optimized)
    #[cfg(feature = "tensorrt")]
    fn try_create_tensorrt_session(
        model_path: &str,
        config: &DetectorConfig,
    ) -> Result<Session, DetectionError> {
        use ort::execution_providers::TensorRTExecutionProvider;

        info!("Attempting to use TensorRT backend (NVIDIA GPU - optimized)...");

        let session = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_execution_providers([TensorRTExecutionProvider::default()
                .with_device_id(config.gpu_device_id)
                // TensorRT will automatically use FP16 if the model is FP16
                .with_fp16(true) // Enable FP16 optimization
                .build()])
            .map_err(|e| {
                DetectionError::ModelLoadError(format!("TensorRT provider failed: {}", e))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                DetectionError::ModelLoadError(format!("Failed to load model with TensorRT: {}", e))
            })?;

        Ok(session)
    }

    /// Try to create a GPU-accelerated session with CoreML
    #[cfg(feature = "metal")]
    fn try_create_gpu_session(
        model_path: &str,
        _config: &DetectorConfig,
    ) -> Result<Session, DetectionError> {
        use ort::execution_providers::CoreMLExecutionProvider;

        info!("Attempting to use CoreML (Metal) backend...");

        let session = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| DetectionError::ModelLoadError(format!("CoreML provider failed: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| {
                DetectionError::ModelLoadError(format!("Failed to load FP16 model: {}", e))
            })?;

        Ok(session)
    }

    /// Create a CPU-only session
    fn create_cpu_session(model_path: &str) -> Result<Session, DetectionError> {
        let session = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| DetectionError::ModelLoadError(format!("Failed to load model: {}", e)))?;

        Ok(session)
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

    /// Detect targets in multiple images using batch inference (parallel GPU execution)
    pub fn detect_batch(
        &mut self,
        images: &[ImageData],
    ) -> Result<Vec<Vec<Detection>>, DetectionError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Starting batch detection on {} images", images.len());
        let batch_size = images.len();

        // Preprocess all images into a single batch tensor [N, 3, 640, 640]
        let mut batch_tensor = Array::zeros((batch_size, 3, 640, 640));

        for (batch_idx, image) in images.iter().enumerate() {
            let preprocessed = self.preprocess(image)?;
            // Copy from [1, 3, 640, 640] to batch position [batch_idx, 3, 640, 640]
            for c in 0..3 {
                for y in 0..640 {
                    for x in 0..640 {
                        batch_tensor[[batch_idx, c, y, x]] = preprocessed[[0, c, y, x]];
                    }
                }
            }
        }

        // Run batch inference
        let batch_tensor_dyn = batch_tensor.into_dyn();
        let tensor_ref = TensorRef::from_array_view(&batch_tensor_dyn)
            .map_err(|e| DetectionError::InferenceError(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| DetectionError::InferenceError(e.to_string()))?;

        // Extract batch output
        let output_array = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| DetectionError::postprocessing(e.to_string()))?
            .into_owned();

        drop(outputs);

        // Process each image's detections from batch output
        let mut all_detections = Vec::with_capacity(batch_size);

        for (batch_idx, image) in images.iter().enumerate() {
            // Extract output for this image from batch
            // YOLOv8 batch output shape: [batch_size, num_channels, num_boxes]
            let shape = output_array.shape();
            if shape.len() != 3 {
                return Err(DetectionError::postprocessing(format!(
                    "Unexpected batch output shape: {:?}",
                    shape
                )));
            }

            // Create view for this image's output: [1, num_channels, num_boxes]
            let image_output = output_array.slice(ndarray::s![batch_idx, .., ..]);
            let image_output = image_output.insert_axis(ndarray::Axis(0));

            // Postprocess this image's detections
            let detections =
                self.postprocess(image_output.view().into_dyn(), image.width, image.height)?;

            all_detections.push(detections);
        }

        info!("Batch detection complete: {} images processed", batch_size);
        Ok(all_detections)
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
