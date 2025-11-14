/// Frame detection pipeline with pre-processing, execution, and post-processing stages
/// 
/// Processes single frames through:
/// 1. Pre-processing: Tile extraction + shadow removal
/// 2. Execution: Frame-by-frame detection via FrameExecutor
/// 3. Post-processing: NMS + coordinate merging
use crate::frame_executor::FrameExecutor;
use crate::tracking_utils::{calculate_iou, BoundingBox};
use crate::types::{ImageData, ImageFormat};
use image::{RgbImage};
use opencv::{
    core::{Mat, Size, Vector, CV_8UC3},
    imgproc,
    prelude::*,
};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::task;

/// Tile with its position in the original image
#[derive(Clone)]
pub struct Tile {
    pub image: RgbImage,
    pub offset_x: u32,
    pub offset_y: u32,
    pub tile_idx: usize,
}

/// Pre-processing output: tiles extracted from original image
#[derive(Clone)]
pub struct PreprocessOutput {
    pub tiles: Vec<Tile>,  // Tiles extracted from deshadowed image (used for detection)
    pub resized_width: u32,
    pub resized_height: u32,
    pub resized_image: RgbImage,  // Original resized image (for annotation/visualization)
}

/// Detection result with tile information
#[derive(Clone, Debug)]
pub struct TileDetection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: String,
    pub tile_idx: usize,
    // Optional Kalman filter velocity (pixels per second)
    pub vx: Option<f32>,
    pub vy: Option<f32>,
    // Optional track ID for persistent object identification
    pub track_id: Option<u32>,
}

/// Execution output: detections from all tiles
#[derive(Clone)]
pub struct ExecutionOutput {
    pub detections: Vec<TileDetection>,
    pub resized_width: u32,
    pub resized_height: u32,
    pub resized_image: RgbImage,
}

/// Final output after post-processing
#[derive(Clone)]
pub struct PipelineOutput {
    pub detections: Vec<TileDetection>,
    pub resized_width: u32,
    pub resized_height: u32,
    pub resized_image: RgbImage,
    pub duplicates_removed: usize,
    pub nested_removed: usize,
    pub pipeline_total_time_ms: f32,
}

impl Default for PipelineOutput {
    fn default() -> Self {
        Self {
            detections: Vec::new(),
            resized_width: 0,
            resized_height: 0,
            resized_image: RgbImage::new(1, 1), // Minimal 1x1 dummy image
            duplicates_removed: 0,
            nested_removed: 0,
            pipeline_total_time_ms: 0.0,
        }
    }
}

/// Pre-processing stage: Extract tiles and apply shadow removal
#[derive(Clone)]
pub struct PreprocessStage {
    tile_size: u32,
    tile_overlap: u32,
}

impl PreprocessStage {
    pub fn new(tile_size: u32, tile_overlap: u32) -> Self {
        Self { tile_size, tile_overlap }
    }

    /// Resize image so the longer dimension fits tile_size while preserving aspect ratio
    /// This ensures the entire image fits within a single tile (no tiling needed)
    fn _resize_to_fit(&self, img: &RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        
        // If both dimensions are already <= tile_size, no need to resize
        if width <= self.tile_size && height <= self.tile_size {
            return img.clone();
        }

        // Calculate scale factor based on the LONGER dimension
        let scale = if width > height {
            self.tile_size as f32 / width as f32
        } else {
            self.tile_size as f32 / height as f32
        };

        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        // OPTIMIZATION: Use Triangle filter (3-5x faster than Lanczos3)
        // Triangle provides good quality for real-time processing
        image::imageops::resize(
            img,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle,
        )
    }

    /// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for shadow removal
    /// Uses OpenCV's CLAHE on the L channel of LAB color space
    fn remove_shadows_clahe(&self, img: &RgbImage) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let width = img.width();
        let height = img.height();
        
        // Convert RgbImage to OpenCV Mat (RGB format)
        let data = img.as_raw();
        let rgb_mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                height as i32,
                width as i32,
                CV_8UC3,
                data.as_ptr() as *mut _,
                opencv::core::Mat_AUTO_STEP,
            )?
        };
        
        // Convert RGB to LAB color space
        let mut lab_mat = Mat::default();
        imgproc::cvt_color(
            &rgb_mat,
            &mut lab_mat,
            imgproc::COLOR_RGB2Lab,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        
        // Split LAB channels
        let mut lab_channels = Vector::<Mat>::new();
        opencv::core::split(&lab_mat, &mut lab_channels)?;
        
        // Apply CLAHE to L channel (lightness)
        let mut clahe = imgproc::create_clahe(2.0, Size::new(8, 8))?;
        let mut l_equalized = Mat::default();
        clahe.apply(&lab_channels.get(0)?, &mut l_equalized)?;
        
        // Replace L channel with equalized version
        lab_channels.set(0, l_equalized)?;
        
        // Merge channels back
        let mut lab_enhanced = Mat::default();
        opencv::core::merge(&lab_channels, &mut lab_enhanced)?;
        
        // Convert back to RGB
        let mut rgb_enhanced = Mat::default();
        imgproc::cvt_color(
            &lab_enhanced,
            &mut rgb_enhanced,
            imgproc::COLOR_Lab2RGB,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        
        // Convert Mat back to RgbImage
        let data_bytes = rgb_enhanced.data_bytes()?.to_vec();
        Ok(RgbImage::from_vec(width, height, data_bytes)
            .ok_or("Failed to create RgbImage from CLAHE result")?)
    }

    /// Calculate tile positions to cover the entire image
    fn get_tile_positions(&self, img_width: u32, img_height: u32) -> Vec<(u32, u32)> {
        let mut positions = Vec::new();

        if img_width <= self.tile_size && img_height <= self.tile_size {
            positions.push((0, 0));
            return positions;
        }

        let stride = self.tile_size.saturating_sub(self.tile_overlap);
        
        // Calculate number of tiles needed to cover the entire image
        let tiles_x = if img_width <= self.tile_size {
            1
        } else {
            // Calculate how many tiles we need
            // First tile covers [0, tile_size]
            // Each subsequent tile covers [i*stride, i*stride + tile_size]
            let remaining_after_first = img_width.saturating_sub(self.tile_size);
            let additional_tiles = (remaining_after_first as f32 / stride as f32).ceil() as u32;
            1 + additional_tiles
        };

        let tiles_y = if img_height <= self.tile_size {
            1
        } else {
            // Calculate how many tiles we need
            let remaining_after_first = img_height.saturating_sub(self.tile_size);
            let additional_tiles = (remaining_after_first as f32 / stride as f32).ceil() as u32;
            1 + additional_tiles
        };

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let x = if tx == tiles_x - 1 && img_width > self.tile_size {
                    img_width.saturating_sub(self.tile_size)
                } else {
                    (tx * stride).min(img_width.saturating_sub(self.tile_size))
                };

                let y = if ty == tiles_y - 1 && img_height > self.tile_size {
                    img_height.saturating_sub(self.tile_size)
                } else {
                    (ty * stride).min(img_height.saturating_sub(self.tile_size))
                };

                positions.push((x, y));
            }
        }

        positions.sort();
        positions.dedup();
        positions
    }

    /// Extract tiles from image (shadow removal already applied to full image)
    /// OPTIMIZATION: Uses parallel iteration for multi-tile extraction
    pub fn extract_tiles(&self, img: &RgbImage) -> Vec<Tile> {
        let positions = self.get_tile_positions(img.width(), img.height());

        positions
            .par_iter() // PARALLEL: Process tiles concurrently
            .enumerate()
            .map(|(idx, &(x, y))| {
                let crop_width = self.tile_size.min(img.width() - x);
                let crop_height = self.tile_size.min(img.height() - y);

                let cropped =
                    image::imageops::crop_imm(img, x, y, crop_width, crop_height).to_image();

                // Pad if necessary
                let tile = if crop_width < self.tile_size || crop_height < self.tile_size {
                    let mut padded = RgbImage::new(self.tile_size, self.tile_size);
                    // OPTIMIZATION: Use fill instead of loop for faster initialization
                    padded.fill(114);  // Gray padding color
                    image::imageops::overlay(&mut padded, &cropped, 0, 0);
                    padded
                } else {
                    cropped
                };

                Tile {
                    image: tile,
                    offset_x: x,
                    offset_y: y,
                    tile_idx: idx,
                }
            })
            .collect()
    }

    /// Run pre-processing stage
    pub async fn process(&self, img: RgbImage) -> PreprocessOutput {
        // Resize image to fit tile_size on longer dimension while preserving aspect ratio
        // This ensures the entire image fits in a single tile
        let resized_img = img.clone(); //self.resize_to_fit(&img);
        let resized_width = resized_img.width();
        let resized_height = resized_img.height();

        // Apply CLAHE-based shadow removal to entire resized image once (more efficient than per-tile)
        // CLAHE improves object detection quality and must always be applied
        let deshadowed_img = match self.remove_shadows_clahe(&resized_img) {
            Ok(enhanced) => enhanced,
            Err(e) => {
                eprintln!("⚠️  CLAHE failed, using original image: {}", e);
                resized_img.clone()
            }
        };

        // Extract tiles from shadow-removed image (tiles will be used for detection)
        // Run tile extraction in a blocking task to avoid blocking the async executor
        let tiles = task::spawn_blocking({
            let deshadowed_img = deshadowed_img.clone();
            let tile_size = self.tile_size;
            let tile_overlap = self.tile_overlap;
            move || {
                let preprocess_stage = PreprocessStage::new(tile_size, tile_overlap);
                preprocess_stage.extract_tiles(&deshadowed_img)
            }
        }).await.unwrap();
        
        PreprocessOutput {
            tiles,
            resized_width,
            resized_height,
            resized_image: resized_img,  // Keep original resized for annotation
        }
    }
}

/// Execution stage: Run detection on tiles using frame executor
#[derive(Clone)]
pub struct ExecutionStage {
    frame_executor: Arc<FrameExecutor>,
    tile_size: u32,
    allowed_classes: Vec<u32>,
}

impl ExecutionStage {
    pub fn new(
        frame_executor: Arc<FrameExecutor>,
        tile_size: u32,
        allowed_classes: Vec<u32>,
    ) -> Self {
        Self {
            frame_executor,
            tile_size,
            allowed_classes,
        }
    }

    /// Run detection on tiles
    pub async fn process(&self, input: PreprocessOutput) -> ExecutionOutput {
        // OPTIMIZATION: Pre-allocate detection vector (typical: 5-15 detections per tile)
        let mut all_detections = Vec::with_capacity(input.tiles.len() * 10);

        // Submit all tiles to batch executor
        let mut responses = Vec::new();
        
        // PARALLEL: Convert all tiles to ImageData concurrently
        let tile_data_vec: Vec<_> = task::spawn_blocking({
            let tiles = input.tiles.clone();
            move || {
                tiles
                    .par_iter()
                    .map(|tile| {
                        // Pre-allocate with exact capacity for better performance
                        let mut tile_data = Vec::with_capacity((tile.image.width() * tile.image.height() * 3) as usize);
                        
                        // Flatten RGB data
                        for pixel in tile.image.pixels() {
                            tile_data.push(pixel[0]);
                            tile_data.push(pixel[1]);
                            tile_data.push(pixel[2]);
                        }

                        ImageData::new(
                            tile_data,
                            tile.image.width(),
                            tile.image.height(),
                            ImageFormat::RGB,
                        )
                    })
                    .collect::<Vec<_>>()
            }
        }).await.unwrap();

        // Submit tiles sequentially (submission must be sequential for ordering)
        for (tile, image_data) in input.tiles.iter().zip(tile_data_vec) {
            let response_rx = self.frame_executor.detect_async(image_data);
            
            // Only add to responses if frame wasn't dropped due to backpressure
            if let Some(rx) = response_rx {
                responses.push((tile.tile_idx, tile.offset_x, tile.offset_y, rx));
            } else {
                log::warn!("⚠️  Tile {} dropped due to detector backpressure", tile.tile_idx);
            }
        }

        // Collect results asynchronously
        let mut tile_results = Vec::new();
        for (tile_idx, offset_x, offset_y, response_rx) in responses {
            // Use tokio's blocking recv since std::sync::mpsc::Receiver is not async
            match task::spawn_blocking(move || response_rx.recv()).await.unwrap() {
                Ok(Ok(detections)) => {
                    tile_results.push((tile_idx, offset_x, offset_y, detections));
                }
                Ok(Err(_)) => {
                    log::warn!("⚠️  Detection failed for tile {}", tile_idx);
                }
                Err(_) => {
                    // Channel disconnected - likely due to shutdown signal
                    log::info!("Detection channel disconnected for tile {} - shutdown signal received", tile_idx);
                    break;
                }
            }
        }

        // PARALLEL: Process all tile detections concurrently
        let processed_detections: Vec<(usize, Vec<TileDetection>)> = task::spawn_blocking({
            let tile_results = tile_results.clone();
            let tile_size = self.tile_size;
            let allowed_classes = self.allowed_classes.clone();
            move || {
                tile_results
                    .par_iter()
                    .map(|(tile_idx, offset_x, offset_y, detections)| {
                        let mut tile_dets = Vec::new();

                        for det in detections {
                            let class_id = det.class.id();

                            // Filter by allowed classes (empty list = allow all)
                            if !allowed_classes.is_empty() && !allowed_classes.contains(&class_id)
                            {
                                continue;
                            }

                            // Convert normalized coordinates to pixels in tile space
                            let x_tile_px = det.bbox.x * tile_size as f32;
                            let y_tile_px = det.bbox.y * tile_size as f32;
                            let w_tile_px = det.bbox.width * tile_size as f32;
                            let h_tile_px = det.bbox.height * tile_size as f32;

                            // Add tile offset to get position in original image
                            let x_final = x_tile_px + *offset_x as f32;
                            let y_final = y_tile_px + *offset_y as f32;

                            // Get class name and capitalize first letter
                            let class_name = det.class.name();
                            let capitalized_name = if !class_name.is_empty() {
                                let mut chars = class_name.chars();
                                match chars.next() {
                                    None => String::new(),
                                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                                }
                            } else {
                                class_name
                            };
                            
                            tile_dets.push(TileDetection {
                                x: x_final,
                                y: y_final,
                                w: w_tile_px,
                                h: h_tile_px,
                                confidence: det.confidence,
                                class_id,
                                class_name: capitalized_name,
                                tile_idx: *tile_idx,
                                vx: None,  // No velocity for raw detections
                                vy: None,
                                track_id: None,  // No track ID yet (assigned by Kalman tracker)
                            });
                        }

                        (*tile_idx, tile_dets)
                    })
                    .collect()
            }
        }).await.unwrap();

        // Collect all detections
        for (_tile_idx, tile_dets) in processed_detections {
            all_detections.extend(tile_dets);
        }
        
        ExecutionOutput {
            detections: all_detections,
            resized_width: input.resized_width,
            resized_height: input.resized_height,
            resized_image: input.resized_image,
        }
    }
}

/// Post-processing stage: Apply NMS and prepare final results
#[derive(Clone)]
pub struct PostprocessStage {
    iou_threshold: f32,
}

impl PostprocessStage {
    pub fn new(iou_threshold: f32) -> Self {
        Self { iou_threshold }
    }

    /// Calculate IoU between two bounding boxes using tracking_utils
    /// OPTIMIZATION: Inline for performance + reuses optimized utility function
    #[inline]
    fn calculate_iou(a: &TileDetection, b: &TileDetection) -> f32 {
        // Convert TileDetection to BoundingBox format
        let box_a = BoundingBox {
            x1: a.x,
            y1: a.y,
            x2: a.x + a.w,
            y2: a.y + a.h,
        };
        
        let box_b = BoundingBox {
            x1: b.x,
            y1: b.y,
            x2: b.x + b.w,
            y2: b.y + b.h,
        };

        // Use optimized utility function
        calculate_iou(&box_a, &box_b)
    }

    /// Check if detection `inner` is fully contained within detection `outer`
    /// with optional padding applied to outer detection boundaries
    #[inline]
    fn is_fully_contained(inner: &TileDetection, outer: &TileDetection, padding: f32) -> bool {
        let inner_x1 = inner.x;
        let inner_y1 = inner.y;
        let inner_x2 = inner.x + inner.w;
        let inner_y2 = inner.y + inner.h;

        // Expand outer detection boundaries by padding (10px on each side = 20px total added to width/height)
        let outer_x1 = (outer.x - padding).max(0.0);
        let outer_y1 = (outer.y - padding).max(0.0);
        let outer_x2 = outer.x + outer.w + padding;
        let outer_y2 = outer.y + outer.h + padding;

        inner_x1 >= outer_x1 && inner_y1 >= outer_y1 && inner_x2 <= outer_x2 && inner_y2 <= outer_y2
    }

    /// Filter out detections that are fully contained within other detections
    /// Only removes nested detections if they are of the same class as the parent
    /// OPTIMIZATION: Parallel filtering - each detection checked concurrently
    fn filter_nested_detections(&self, detections: Vec<TileDetection>) -> Vec<TileDetection> {
        const NESTED_PADDING: f32 = 10.0; // Add 10px padding to nested detection boundaries

        // PARALLEL: Check each detection concurrently for containment
        let filtered: Vec<TileDetection> = detections
            .par_iter()
            .enumerate()
            .filter_map(|(i, det)| {
                // Check if this detection is contained in any other detection
                let is_contained = detections.iter().enumerate().any(|(j, other)| {
                    i != j 
                        && det.class_id == other.class_id  // Only remove if same class
                        && Self::is_fully_contained(det, other, NESTED_PADDING)
                });

                if !is_contained {
                    Some(det.clone())
                } else {
                    None
                }
            })
            .collect();

        filtered
    }

    /// Apply NMS to remove duplicates
    /// OPTIMIZATION: Parallel NMS by class - each class processed independently
    /// OPTIMIZATION: Parallel IoU calculation - suppression checks run concurrently
    fn apply_nms(&self, detections: Vec<TileDetection>) -> Vec<TileDetection> {
        if detections.is_empty() {
            return detections;
        }

        // PARALLEL: Group detections by class using concurrent collection
        let classes: HashSet<u32> = detections.par_iter().map(|d| d.class_id).collect();
        
        // PARALLEL: Process each class independently
        let results: Vec<TileDetection> = classes
            .par_iter()
            .flat_map(|&class_id| {
                // Get all detections for this class
                let mut class_dets: Vec<TileDetection> = detections
                    .iter()
                    .filter(|d| d.class_id == class_id)
                    .cloned()
                    .collect();

                // Sort by confidence (highest first)
                class_dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

                // Apply NMS for this class
                let mut keep = Vec::new();
                let mut suppressed = vec![false; class_dets.len()];

                for i in 0..class_dets.len() {
                    if suppressed[i] {
                        continue;
                    }

                    keep.push(class_dets[i].clone());

                    // PARALLEL: Calculate IoU for all remaining detections concurrently
                    // Collect indices that should be suppressed
                    let to_suppress: Vec<usize> = (i + 1..class_dets.len())
                        .into_par_iter()
                        .filter(|&j| {
                            !suppressed[j] 
                                && Self::calculate_iou(&class_dets[i], &class_dets[j]) > self.iou_threshold
                        })
                        .collect();

                    // Mark all suppressed indices
                    for j in to_suppress {
                        suppressed[j] = true;
                    }
                }

                keep
            })
            .collect();

        results
    }

    /// Run post-processing stage
    pub async fn process<F>(&self, input: ExecutionOutput, callback: F, pipeline_start_time: Instant) -> PipelineOutput
    where
        F: FnOnce(&PipelineOutput),
    {
        let before_nms = input.detections.len();

        // Step 1: Apply NMS to remove duplicates based on IoU
        let after_nms = task::spawn_blocking({
            let detections = input.detections.clone();
            let iou_threshold = self.iou_threshold;
            move || {
                let postprocess_stage = PostprocessStage::new(iou_threshold);
                postprocess_stage.apply_nms(detections)
            }
        }).await.unwrap();
        let duplicates_removed = before_nms - after_nms.len();

        // Step 2: Filter nested detections (fully contained boxes)
        let before_nested_filter = after_nms.len();
        let filtered_detections = task::spawn_blocking({
            let after_nms = after_nms.clone();
            let iou_threshold = self.iou_threshold;
            move || {
                let postprocess_stage = PostprocessStage::new(iou_threshold);
                postprocess_stage.filter_nested_detections(after_nms)
            }
        }).await.unwrap();
        let nested_removed = before_nested_filter - filtered_detections.len();

        // Calculate total pipeline time
        let pipeline_total_time_ms = pipeline_start_time.elapsed().as_secs_f32() * 1000.0;

        let detections_count = filtered_detections.len();

        let output = PipelineOutput {
            detections: filtered_detections,
            resized_width: input.resized_width,
            resized_height: input.resized_height,
            resized_image: input.resized_image,
            duplicates_removed,
            nested_removed,
            pipeline_total_time_ms,
        };

        log::info!(
            "{} detected, {} filtered out, time: {:.2} ms",
            detections_count,
            duplicates_removed + nested_removed,
            pipeline_total_time_ms,
        );
        // Execute callback with final pipeline output
        callback(&output);

        output
    }
}

/// Detection pipeline configuration
pub struct PipelineConfig {
    pub tile_overlap: u32,
    pub allowed_classes: Vec<u32>,
    pub iou_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tile_overlap: 64,
            allowed_classes: vec![2, 3, 4, 7], // car, motorcycle, airplane, truck
            iou_threshold: 0.5,
        }
    }
}

/// Detection pipeline that processes a single image through three stages
/// The detector pool is shared across multiple pipeline executions
#[derive(Clone)]
pub struct DetectionPipeline {
    preprocess: PreprocessStage,
    execution: ExecutionStage,
    postprocess: PostprocessStage,
}

impl DetectionPipeline {
    /// Create a new detection pipeline with frame executor
    pub fn new(frame_executor: Arc<FrameExecutor>, config: PipelineConfig) -> Self {
        // Get tile size from detector's input size
        let (tile_size, _) = frame_executor.input_size();

        let preprocess = PreprocessStage::new(tile_size, config.tile_overlap);
        let execution = ExecutionStage::new(frame_executor, tile_size, config.allowed_classes);
        let postprocess = PostprocessStage::new(config.iou_threshold);

        Self {
            preprocess,
            execution,
            postprocess,
        }
    }

    pub async fn process_with_callback_local<F>(&self, img: RgbImage, callback: F)
    where
        F: FnOnce(&PipelineOutput),
    {
        let pipeline_start_time = Instant::now();
        
        // Stage 1: Preprocess (await directly)
        let preprocess_result = self.preprocess.process(img).await;

        // Stage 2: Execution
        let execution_result = self.execution.process(preprocess_result).await;

        // Stage 3: Postprocess
        self.postprocess.process(execution_result, callback, pipeline_start_time).await;
    }

    /// Synchronous version of process_with_callback for non-tokio contexts
    /// This method runs all processing synchronously and calls the callback directly
    pub fn process_with_callback<F>(&self, img: &RgbImage, callback: F)
    where
        F: FnOnce(&PipelineOutput),
    {
        let img_clone = img.clone();
        let pipeline_start_time = Instant::now();
        
        // Create a new runtime for this synchronous context
        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        
        rt.block_on(async {
            // Stage 1: Preprocess
            let preprocess_result = self.preprocess.process(img_clone).await;

            // Stage 2: Execution
            let execution_result = self.execution.process(preprocess_result).await;

            // Stage 3: Postprocess
            let _output = self.postprocess.process(execution_result, callback, pipeline_start_time).await;
        });
    }

    /// Async fire-and-forget wrapper for process_with_callback 
    /// This method starts the async processing and immediately returns without waiting
    /// The callback will be executed asynchronously when processing completes
    pub async fn process_with_callback_async<F>(&self, img: &RgbImage, callback: F)
    where
        // This async wrapper accepts non-'static callbacks (no Send bound).
        F: FnOnce(&PipelineOutput),
    {
        let img_clone = img.clone();
        
        // Run the processing directly without spawning (caller handles async context)
        self.process_with_callback_local(img_clone, callback).await;
    }

}
