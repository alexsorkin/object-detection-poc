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
use ndarray::{Array, ArrayView, IxDyn};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use rayon::prelude::*;

/// RT-DETR detector for object detection
pub struct RTDETRDetector {
    session: Session,
    config: DetectorConfig,
    input_size: (u32, u32), // Model-specific input size (640x640, same as YOLO)
}

impl RTDETRDetector {
    /// Create a new RT-DETR detector with automatic GPU/CPU selection
    pub fn new(config: DetectorConfig) -> Result<Self, DetectionError> {
        log::info!("Initializing RT-DETR detector");

        // RT-DETR uses 640x640 input (same as YOLO for Ultralytics version)
        let input_size = (640, 640);
        log::info!("RT-DETR input size: {}x{}", input_size.0, input_size.1);

        // Determine which model to use
        let (model_path, use_gpu_backend) = Self::select_model(&config)?;
        log::info!("Selected model: {}", model_path);

        // Try GPU path first if requested
        if config.use_gpu && use_gpu_backend {
            // Try CUDA first (NVIDIA)
            #[cfg(feature = "cuda")]
            {
                match Self::try_create_cuda_session(&model_path, &config) {
                    Ok(session) => {
                        log::info!("✓ RT-DETR loaded successfully with CUDA (NVIDIA GPU)");
                        return Ok(Self {
                            session,
                            config,
                            input_size,
                        });
                    }
                    Err(e) => {
                        log::warn!("CUDA initialization failed: {}", e);
                        log::warn!("Trying TensorRT fallback...");
                    }
                }
            }

            // Try TensorRT (NVIDIA - optimized)
            #[cfg(feature = "tensorrt")]
            {
                match Self::try_create_tensorrt_session(&model_path, &config) {
                    Ok(session) => {
                        log::info!("✓ RT-DETR loaded successfully with TensorRT (NVIDIA GPU)");
                        return Ok(Self {
                            session,
                            config,
                            input_size,
                        });
                    }
                    Err(e) => {
                        log::warn!("TensorRT initialization failed: {}", e);
                    }
                }
            }

            // Try CoreML (Apple Metal)
            #[cfg(feature = "coreml")]
            {
                match Self::try_create_coreml_session(&model_path, &config) {
                    Ok(session) => {
                        log::info!("✓ RT-DETR loaded successfully with CoreML (GPU/Metal)");
                        return Ok(Self {
                            session,
                            config,
                            input_size,
                        });
                    }
                    Err(e) => {
                        log::warn!("CoreML initialization failed: {}", e);
                        log::warn!("Falling back to next GPU backend or CPU...");
                    }
                }
            }

            // Try OpenVINO (Intel GPU/VPU)
            #[cfg(feature = "openvino")]
            {
                match Self::try_create_openvino_session(&model_path, &config) {
                    Ok(session) => {
                        log::info!("✓ RT-DETR loaded successfully with OpenVINO (Intel GPU)");
                        return Ok(Self {
                            session,
                            config,
                            input_size,
                        });
                    }
                    Err(e) => {
                        log::warn!("OpenVINO initialization failed: {}", e);
                        log::warn!("Falling back to CPU with FP32 model...");
                    }
                }
            }

            #[cfg(not(any(
                feature = "coreml",
                feature = "cuda",
                feature = "tensorrt",
                feature = "openvino"
            )))]
            {
                log::warn!("No GPU backend feature enabled, falling back to CPU");
            }
        }

        // CPU fallback path
        let fallback_model = Self::get_fallback_model(&config)?;
        log::info!("Using CPU fallback model: {}", fallback_model);

        let session = Self::create_cpu_session(&fallback_model)?;
        log::info!("✓ RT-DETR loaded successfully with CPU");

        Ok(Self {
            session,
            config,
            input_size,
        })
    }

    /// Select appropriate model based on config
    fn select_model(config: &DetectorConfig) -> Result<(String, bool), DetectionError> {
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

        log::info!("Attempting to use CUDA backend (NVIDIA GPU)...");

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

        log::info!("Attempting to use TensorRT backend (NVIDIA GPU - optimized)...");

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
    #[cfg(feature = "coreml")]
    fn try_create_coreml_session(
        model_path: &str,
        _config: &DetectorConfig,
    ) -> Result<Session, DetectionError> {
        use ort::execution_providers::coreml::CoreMLComputeUnits;
        use ort::execution_providers::CoreMLExecutionProvider;

        log::info!("Attempting to use CoreML (Metal GPU only, no Neural Engine)...");

        let session = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_execution_providers([CoreMLExecutionProvider::default()
                .with_compute_units(CoreMLComputeUnits::CPUAndGPU) // Use GPU only, disable Neural Engine
                .build()])
            .map_err(|e| DetectionError::ModelLoadError(format!("CoreML provider failed: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| DetectionError::ModelLoadError(format!("Failed to load model: {}", e)))?;

        Ok(session)
    }

    /// Try to create a GPU-accelerated session with OpenVINO
    #[cfg(feature = "openvino")]
    fn try_create_openvino_session(
        model_path: &str,
        _config: &DetectorConfig,
    ) -> Result<Session, DetectionError> {
        use ort::execution_providers::OpenVINOExecutionProvider;

        log::info!("Attempting to use OpenVINO backend (Intel GPU/VPU)...");

        let session = Session::builder()
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectionError::ModelLoadError(e.to_string()))?
            .with_execution_providers([OpenVINOExecutionProvider::default().build()])
            .map_err(|e| {
                DetectionError::ModelLoadError(format!("OpenVINO provider failed: {}", e))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                DetectionError::ModelLoadError(format!("Failed to load model with OpenVINO: {}", e))
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
        log::debug!(
            "Starting RT-DETR detection on {}x{} image",
            image.width,
            image.height
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

            // PARALLEL: Split combined tensor using ndarray parallel iteration
            // Pre-compute box and logit slices for each batch
            let boxes_logits: Vec<_> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    let mut batch_boxes = Array::zeros((num_queries, 4));
                    let mut batch_logits = Array::zeros((num_queries, num_classes));

                    for q in 0..num_queries {
                        for i in 0..4 {
                            batch_boxes[[q, i]] = combined_output[[b, q, i]];
                        }
                        for i in 0..num_classes {
                            batch_logits[[q, i]] = combined_output[[b, q, 4 + i]];
                        }
                    }
                    (batch_boxes, batch_logits)
                })
                .collect();

            // Reassemble into final arrays
            let mut pred_boxes = Array::zeros((batch_size, num_queries, 4));
            let mut pred_logits = Array::zeros((batch_size, num_queries, num_classes));

            for (b, (boxes, logits)) in boxes_logits.into_iter().enumerate() {
                pred_boxes.slice_mut(ndarray::s![b, .., ..]).assign(&boxes);
                pred_logits
                    .slice_mut(ndarray::s![b, .., ..])
                    .assign(&logits);
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

        log::debug!("Detected {} targets", detections.len());
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

        log::debug!(
            "Starting RT-DETR batch detection on {} images",
            images.len()
        );
        let batch_size = images.len();

        // PARALLEL: Preprocess all images concurrently
        let preprocessed_images: Vec<_> = images
            .par_iter()
            .map(|image| self.preprocess(image))
            .collect::<Result<Vec<_>, _>>()?;

        // Assemble into batch tensor [N, 3, 640, 640]
        let mut batch_tensor = Array::zeros((
            batch_size,
            3,
            self.input_size.1 as usize,
            self.input_size.0 as usize,
        ));

        for (batch_idx, preprocessed) in preprocessed_images.iter().enumerate() {
            batch_tensor
                .slice_mut(ndarray::s![batch_idx, .., .., ..])
                .assign(&preprocessed.slice(ndarray::s![0, .., .., ..]));
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

            // PARALLEL: Split combined tensor for each batch
            let boxes_logits: Vec<_> = (0..actual_batch_size)
                .into_par_iter()
                .map(|b| {
                    let mut batch_boxes = Array::zeros((num_queries, 4));
                    let mut batch_logits = Array::zeros((num_queries, num_classes));

                    for q in 0..num_queries {
                        for i in 0..4 {
                            batch_boxes[[q, i]] = combined_output[[b, q, i]];
                        }
                        for i in 0..num_classes {
                            batch_logits[[q, i]] = combined_output[[b, q, 4 + i]];
                        }
                    }
                    (batch_boxes, batch_logits)
                })
                .collect();

            // Reassemble into final arrays
            let mut pred_boxes = Array::zeros((actual_batch_size, num_queries, 4));
            let mut pred_logits = Array::zeros((actual_batch_size, num_queries, num_classes));

            for (b, (boxes, logits)) in boxes_logits.into_iter().enumerate() {
                pred_boxes.slice_mut(ndarray::s![b, .., ..]).assign(&boxes);
                pred_logits
                    .slice_mut(ndarray::s![b, .., ..])
                    .assign(&logits);
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

        log::info!("Batch detection complete: {} images processed", batch_size);
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

        // Resize to 640x640 (RT-DETR input size)
        let (target_w, target_h) = self.input_size;
        let resized = img.resize_exact(target_w, target_h, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // PARALLEL: Convert to normalized CHW format [1, 3, 640, 640]
        // Process rows in parallel and collect results
        let pixel_rows: Vec<Vec<[f32; 3]>> = (0..target_h as usize)
            .into_par_iter()
            .map(|y| {
                (0..target_w as usize)
                    .map(|x| {
                        let pixel = rgb.get_pixel(x as u32, y as u32);
                        [
                            pixel[0] as f32 / 255.0,
                            pixel[1] as f32 / 255.0,
                            pixel[2] as f32 / 255.0,
                        ]
                    })
                    .collect()
            })
            .collect();

        // Assemble into final array
        let mut input_data = Array::zeros((1, 3, target_h as usize, target_w as usize));
        for (y, row) in pixel_rows.iter().enumerate() {
            for (x, pixel) in row.iter().enumerate() {
                input_data[[0, 0, y, x]] = pixel[0];
                input_data[[0, 1, y, x]] = pixel[1];
                input_data[[0, 2, y, x]] = pixel[2];
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

        log::debug!(
            "Processing {} queries with {} classes",
            num_queries,
            num_classes
        );

        let mut detections = Vec::new();

        // Process each query
        for i in 0..num_queries {
            // Extract box coordinates (cx, cy, w, h) - normalized 0-1
            let cx = pred_boxes[[0, i, 0]];
            let cy = pred_boxes[[0, i, 1]];
            let w = pred_boxes[[0, i, 2]];
            let h = pred_boxes[[0, i, 3]];

            // Find class with highest score and apply softmax to convert logits to probabilities
            let mut logits = Vec::with_capacity(num_classes);
            let mut max_logit = f32::NEG_INFINITY;

            // Collect logits and find max for numerical stability
            for c in 0..num_classes {
                let logit = pred_logits[[0, i, c]];
                logits.push(logit);
                if logit > max_logit {
                    max_logit = logit;
                }
            }

            // Apply softmax: exp(logit - max_logit) / sum(exp(logit - max_logit))
            let mut exp_sum = 0.0_f32;
            for logit in &logits {
                exp_sum += (logit - max_logit).exp();
            }

            // Find class with highest probability
            let mut max_score = f32::NEG_INFINITY;
            let mut max_class = 0;
            for c in 0..num_classes {
                let prob = (logits[c] - max_logit).exp() / exp_sum;
                if prob > max_score {
                    max_score = prob;
                    max_class = c;
                }
            }

            // max_score is now a proper probability (0-1) after softmax
            let confidence = max_score;

            // Filter by confidence threshold
            if confidence < self.config.confidence_threshold {
                continue;
            }

            log::debug!(
                "Accepted detection: class={}, conf={:.3}, threshold={:.3}",
                max_class,
                confidence,
                self.config.confidence_threshold
            );

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

        log::debug!("Found {} detections above threshold", detections.len());

        // Note: No NMS needed for RT-DETR! The transformer already outputs unique detections
        Ok(detections)
    }
}
