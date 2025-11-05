//! RT-DETR (Real-time DEtection TRansformer) detector implementation
//!
//! RT-DETR advantages over YOLO:
//! - No NMS required (transformer outputs unique detections)
//! - Better accuracy for small objects
//! - End-to-end detection with 300 queries vs 8400 YOLO anchors
//!
//! Model outputs:
//! - pred_boxes: [batch, 300, 4] - (cx, cy, w, h) normalized 0-1
//! - pred_logits: [batch, 300, num_classes] - class scores (logits, not softmax)

use crate::error::DetectionError;
use crate::types::{BoundingBox, Detection, DetectorConfig, ImageData, TargetClass};
use log::{debug, info, warn};
use ndarray::{Array, ArrayView, IxDyn};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

/// RT-DETR detector for object detection
pub struct RTDETRDetector {
    session: Session,
    config: DetectorConfig,
    input_size: (u32, u32), // Model-specific input size (640x640, same as YOLO)
}

impl RTDETRDetector {
    /// Create a new RT-DETR detector with automatic GPU/CPU selection
    pub fn new(config: DetectorConfig) -> Result<Self, DetectionError> {
        info!("Initializing RT-DETR detector");

        // RT-DETR uses 640x640 input (same as YOLO for Ultralytics version)
        let input_size = (640, 640);
        info!("RT-DETR input size: {}x{}", input_size.0, input_size.1);

        // Determine which model to use
        let (model_path, use_gpu_backend) = Self::select_model(&config)?;
        info!("Selected model: {}", model_path);

        // Try GPU path first if requested
        if config.use_gpu && use_gpu_backend {
            // Try CUDA first (NVIDIA)
            #[cfg(feature = "cuda")]
            {
                match Self::try_create_cuda_session(&model_path, &config) {
                    Ok(session) => {
                        info!("✓ RT-DETR loaded successfully with CUDA (NVIDIA GPU)");
                        return Ok(Self {
                            session,
                            config,
                            input_size,
                        });
                    }
                    Err(e) => {
                        warn!("CUDA initialization failed: {}", e);
                        warn!("Trying TensorRT fallback...");
                    }
                }
            }

            // Try TensorRT (NVIDIA - optimized)
            #[cfg(feature = "tensorrt")]
            {
                match Self::try_create_tensorrt_session(&model_path, &config) {
                    Ok(session) => {
                        info!("✓ RT-DETR loaded successfully with TensorRT (NVIDIA GPU)");
                        return Ok(Self {
                            session,
                            config,
                            input_size,
                        });
                    }
                    Err(e) => {
                        warn!("TensorRT initialization failed: {}", e);
                    }
                }
            }

            // Try CoreML (Apple Metal)
            #[cfg(feature = "metal")]
            {
                match Self::try_create_gpu_session(&model_path, &config) {
                    Ok(session) => {
                        info!("✓ RT-DETR loaded successfully with CoreML (GPU/Metal)");
                        return Ok(Self {
                            session,
                            config,
                            input_size,
                        });
                    }
                    Err(e) => {
                        warn!("CoreML initialization failed: {}", e);
                        warn!("Falling back to CPU with FP32 model...");
                    }
                }
            }

            #[cfg(not(any(feature = "metal", feature = "cuda", feature = "tensorrt")))]
            {
                warn!("No GPU backend feature enabled, falling back to CPU");
            }
        }

        // CPU fallback path
        let fallback_model = Self::get_fallback_model(&config)?;
        info!("Using CPU fallback model: {}", fallback_model);

        let session = Self::create_cpu_session(&fallback_model)?;
        info!("✓ RT-DETR loaded successfully with CPU");

        Ok(Self {
            session,
            config,
            input_size,
        })
    }

    /// Select appropriate model based on config
    fn select_model(config: &DetectorConfig) -> Result<(String, bool), DetectionError> {
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

    /// Try to create a GPU-accelerated session with CUDA
    #[cfg(feature = "cuda")]
    fn try_create_cuda_session(
        model_path: &str,
        _config: &DetectorConfig,
    ) -> Result<Session, DetectionError> {
        use ort::execution_providers::CUDAExecutionProvider;

        info!("Attempting to use CUDA backend (NVIDIA GPU)...");

        let session = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| DetectionError::ModelLoadError(format!("CUDA provider failed: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| {
                DetectionError::ModelLoadError(format!("Failed to load model with CUDA: {}", e))
            })?;

        Ok(session)
    }

    /// Try to create a GPU-accelerated session with TensorRT
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
                .with_fp16(true)
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
            .map_err(|e| DetectionError::ModelLoadError(format!("Failed to load model: {}", e)))?;

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
            "Starting RT-DETR detection on {}x{} image",
            image.width, image.height
        );

        // Preprocess: resize to 576x576, normalize, CHW format
        let input_tensor = self.preprocess(image)?;

        // Run inference
        let tensor_ref = TensorRef::from_array_view(&input_tensor)
            .map_err(|e| DetectionError::InferenceError(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| DetectionError::InferenceError(e.to_string()))?;

        // Auto-detect output format: Ultralytics (1 output) vs Hugging Face (2 outputs)
        let (pred_boxes, pred_logits) = if outputs.len() >= 2 {
            // Hugging Face format: separate pred_logits and pred_boxes tensors
            // outputs[0]: pred_logits [1, 300, 80]
            // outputs[1]: pred_boxes [1, 300, 4]
            let logits = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| DetectionError::postprocessing(e.to_string()))?
                .into_owned();
            
            let boxes = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| DetectionError::postprocessing(e.to_string()))?
                .into_owned();
            
            (boxes, logits)
        } else {
            // Ultralytics format: combined [1, 300, 84] where 84 = 4 (bbox) + 80 (classes)
            let combined_output = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| DetectionError::postprocessing(e.to_string()))?
                .into_owned();

            let shape = combined_output.shape();
            let batch_size = shape[0];
            let num_queries = shape[1];
            let total_dim = shape[2];
            let num_classes = total_dim - 4;
            
            let mut pred_boxes = Array::zeros((batch_size, num_queries, 4));
            let mut pred_logits = Array::zeros((batch_size, num_queries, num_classes));
            
            for b in 0..batch_size {
                for q in 0..num_queries {
                    for i in 0..4 {
                        pred_boxes[[b, q, i]] = combined_output[[b, q, i]];
                    }
                    for i in 0..num_classes {
                        pred_logits[[b, q, i]] = combined_output[[b, q, 4 + i]];
                    }
                }
            }
            
            (pred_boxes.into_dyn(), pred_logits.into_dyn())
        };

        drop(outputs);

        // Postprocess: parse RT-DETR output (no NMS needed!)
        let detections = self.postprocess(
            pred_boxes.view(),
            pred_logits.view(),
            image.width,
            image.height,
        )?;

        info!("Detected {} targets", detections.len());
        Ok(detections)
    }

    /// Detect targets in multiple images using batch inference
    pub fn detect_batch(
        &mut self,
        images: &[ImageData],
    ) -> Result<Vec<Vec<Detection>>, DetectionError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "Starting RT-DETR batch detection on {} images",
            images.len()
        );
        let batch_size = images.len();

        // Preprocess all images into a single batch tensor [N, 3, 576, 576]
        let mut batch_tensor = Array::zeros((
            batch_size,
            3,
            self.input_size.1 as usize,
            self.input_size.0 as usize,
        ));

        for (batch_idx, image) in images.iter().enumerate() {
            let preprocessed = self.preprocess(image)?;
            // Copy from [1, 3, 576, 576] to batch position [batch_idx, 3, 576, 576]
            for c in 0..3 {
                for y in 0..self.input_size.1 as usize {
                    for x in 0..self.input_size.0 as usize {
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

        // Auto-detect output format: Ultralytics (1 output) vs Hugging Face (2 outputs)
        let (pred_boxes, pred_logits) = if outputs.len() >= 2 {
            // Hugging Face format: separate pred_logits and pred_boxes tensors
            let logits = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| DetectionError::postprocessing(e.to_string()))?
                .into_owned();
            
            let boxes = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| DetectionError::postprocessing(e.to_string()))?
                .into_owned();
            
            (boxes, logits)
        } else {
            // Ultralytics format: combined [batch, 300, 84]
            let combined_output = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| DetectionError::postprocessing(e.to_string()))?
                .into_owned();

            let shape = combined_output.shape();
            let actual_batch_size = shape[0];
            let num_queries = shape[1];
            let total_dim = shape[2];
            let num_classes = total_dim - 4;
            
            let mut pred_boxes = Array::zeros((actual_batch_size, num_queries, 4));
            let mut pred_logits = Array::zeros((actual_batch_size, num_queries, num_classes));
            
            for b in 0..actual_batch_size {
                for q in 0..num_queries {
                    for i in 0..4 {
                        pred_boxes[[b, q, i]] = combined_output[[b, q, i]];
                    }
                    for i in 0..num_classes {
                        pred_logits[[b, q, i]] = combined_output[[b, q, 4 + i]];
                    }
                }
            }
            
            (pred_boxes.into_dyn(), pred_logits.into_dyn())
        };

        drop(outputs);

        // Process each image's detections from batch output
        let mut all_detections = Vec::with_capacity(batch_size);

        for (batch_idx, image) in images.iter().enumerate() {
            // Extract this image's output: [300, 4] and [300, 80]
            let image_boxes = pred_boxes.slice(ndarray::s![batch_idx, .., ..]);
            let image_logits = pred_logits.slice(ndarray::s![batch_idx, .., ..]);

            // Add batch dimension back for postprocess: [1, 300, 4] and [1, 300, 91]
            let image_boxes = image_boxes.insert_axis(ndarray::Axis(0));
            let image_logits = image_logits.insert_axis(ndarray::Axis(0));

            // Postprocess this image's detections
            let detections = self.postprocess(
                image_boxes.view().into_dyn(),
                image_logits.view().into_dyn(),
                image.width,
                image.height,
            )?;

            all_detections.push(detections);
        }

        info!("Batch detection complete: {} images processed", batch_size);
        Ok(all_detections)
    }

    /// Preprocess image: resize to 576x576, normalize, convert to CHW format
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

        // Resize to 576x576 (RT-DETR input size)
        let (target_w, target_h) = self.input_size;
        let resized = img.resize_exact(target_w, target_h, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // Convert to normalized CHW format [1, 3, 576, 576]
        let mut input_data = Array::zeros((1, 3, target_h as usize, target_w as usize));

        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input_data[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                input_data[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                input_data[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }

        Ok(input_data.into_dyn())
    }

    /// Postprocess RT-DETR output (No NMS needed!)
    ///
    /// RT-DETR outputs:
    /// - pred_boxes: [batch, 300, 4] - (cx, cy, w, h) normalized 0-1
    /// - pred_logits: [batch, 300, 91] - class scores (logits, not probabilities)
    fn postprocess(
        &self,
        pred_boxes: ArrayView<f32, IxDyn>,
        pred_logits: ArrayView<f32, IxDyn>,
        orig_w: u32,
        orig_h: u32,
    ) -> Result<Vec<Detection>, DetectionError> {
        let boxes_shape = pred_boxes.shape();
        let logits_shape = pred_logits.shape();

        // Validate shapes
        if boxes_shape.len() != 3 || boxes_shape[2] != 4 {
            return Err(DetectionError::postprocessing(format!(
                "Unexpected pred_boxes shape: {:?}",
                boxes_shape
            )));
        }

        if logits_shape.len() != 3 {
            return Err(DetectionError::postprocessing(format!(
                "Unexpected pred_logits shape: {:?}",
                logits_shape
            )));
        }

        let num_queries = boxes_shape[1]; // Should be 300
        let num_classes = logits_shape[2]; // Should be 91 for COCO

        debug!(
            "Processing {} queries with {} classes",
            num_queries, num_classes
        );

        let mut detections = Vec::new();

        // Process each query
        for i in 0..num_queries {
            // Extract box coordinates (cx, cy, w, h) - normalized 0-1
            let cx = pred_boxes[[0, i, 0]];
            let cy = pred_boxes[[0, i, 1]];
            let w = pred_boxes[[0, i, 2]];
            let h = pred_boxes[[0, i, 3]];

            // Find class with highest score (apply softmax to logits)
            let mut max_score = f32::NEG_INFINITY;
            let mut max_class = 0;

            for c in 0..num_classes {
                let logit = pred_logits[[0, i, c]];
                if logit > max_score {
                    max_score = logit;
                    max_class = c;
                }
            }

            // Convert logit to probability (simplified - just use sigmoid)
            let confidence = 1.0 / (1.0 + (-max_score).exp());

            // Filter by confidence threshold
            if confidence < self.config.confidence_threshold {
                continue;
            }

            // Convert from (cx, cy, w, h) normalized to absolute pixels
            let x_center = cx * orig_w as f32;
            let y_center = cy * orig_h as f32;
            let width = w * orig_w as f32;
            let height = h * orig_h as f32;

            // Convert to (x, y, w, h) where (x,y) is top-left
            let x = (x_center - width / 2.0).max(0.0);
            let y = (y_center - height / 2.0).max(0.0);

            // Normalize to [0, 1]
            let norm_x = x / orig_w as f32;
            let norm_y = y / orig_h as f32;
            let norm_w = (width / orig_w as f32).min(1.0 - norm_x);
            let norm_h = (height / orig_h as f32).min(1.0 - norm_y);

            let bbox = BoundingBox::new(norm_x, norm_y, norm_w, norm_h);

            // Convert COCO class ID to TargetClass
            let class = TargetClass::from_id(max_class as u32)
                .unwrap_or(TargetClass::Class(max_class as u32));

            detections.push(Detection::new(class, confidence, bbox));
        }

        debug!("Found {} detections above threshold", detections.len());

        // Note: No NMS needed for RT-DETR! The transformer already outputs unique detections
        Ok(detections)
    }
}
