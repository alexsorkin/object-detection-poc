/// Frame detection pipeline with pre-processing, execution, and post-processing stages
/// 
/// Processes single frames through:
/// 1. Pre-processing: Tile extraction + shadow removal
/// 2. Execution: Frame-by-frame detection via FrameExecutor
/// 3. Post-processing: NMS + coordinate merging
use crate::frame_executor::FrameExecutor;
use crate::types::{ImageData, ImageFormat};
use image::{Rgb, RgbImage};
use opencv::{
    core::{Mat, Size, Vector, CV_8UC3},
    imgproc,
    prelude::*,
};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::oneshot;
use tokio::task;

/// Stage result that flows through the pipeline
#[derive(Clone)]
pub struct StageResult<T> {
    pub data: T,
    pub stage_name: String,
    pub duration_ms: f32,
}

/// Timing information for each pipeline stage
#[derive(Clone, Debug)]
pub struct PipelineTiming {
    pub preprocess_ms: f32,
    pub execution_ms: f32,
    pub postprocess_ms: f32,
    pub total_ms: f32,
}

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
    pub original_width: u32,
    pub original_height: u32,
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
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub resized_image: RgbImage,
}

/// Final output after post-processing
#[derive(Clone)]
pub struct PostprocessOutput {
    pub detections: Vec<TileDetection>,
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub resized_image: RgbImage,
    pub duplicates_removed: usize,
    pub nested_removed: usize,
}

/// Pre-processing stage: Extract tiles and apply shadow removal
#[derive(Clone)]
pub struct PreprocessStage {
    tile_size: u32,
    overlap: u32,
}

impl PreprocessStage {
    pub fn new(tile_size: u32, overlap: u32) -> Self {
        Self { tile_size, overlap }
    }

    /// Resize image so the longer dimension fits tile_size while preserving aspect ratio
    /// This ensures the entire image fits within a single tile (no tiling needed)
    fn resize_to_fit(&self, img: &RgbImage) -> RgbImage {
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

    /// Apply HSV-based shadow removal (brightness enhancement) - DEPRECATED: Use CLAHE instead
    #[allow(dead_code)]
    fn remove_shadows_hsv(&self, img: &RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        let mut result = RgbImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let [r, g, b] = pixel.0;

                // RGB to HSV conversion
                let r_f = r as f32 / 255.0;
                let g_f = g as f32 / 255.0;
                let b_f = b as f32 / 255.0;

                let max = r_f.max(g_f).max(b_f);
                let min = r_f.min(g_f).min(b_f);
                let delta = max - min;

                // Calculate H (hue)
                let h = if delta < 0.00001 {
                    0.0
                } else if (max - r_f).abs() < 0.00001 {
                    60.0 * (((g_f - b_f) / delta) % 6.0)
                } else if (max - g_f).abs() < 0.00001 {
                    60.0 * (((b_f - r_f) / delta) + 2.0)
                } else {
                    60.0 * (((r_f - g_f) / delta) + 4.0)
                };

                // Calculate S (saturation)
                let s = if max < 0.00001 { 0.0 } else { delta / max };

                // V (value/brightness)
                let v = max;

                // Enhance V channel for dark pixels (shadows)
                let v_enhanced = if v < 0.5 {
                    (v + 0.3).min(1.0)
                } else {
                    (v + 0.1).min(1.0)
                };

                // HSV back to RGB
                let c = v_enhanced * s;
                let x_val = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
                let m = v_enhanced - c;

                let (r_out, g_out, b_out) = if h < 60.0 {
                    (c, x_val, 0.0)
                } else if h < 120.0 {
                    (x_val, c, 0.0)
                } else if h < 180.0 {
                    (0.0, c, x_val)
                } else if h < 240.0 {
                    (0.0, x_val, c)
                } else if h < 300.0 {
                    (x_val, 0.0, c)
                } else {
                    (c, 0.0, x_val)
                };

                let r_final = ((r_out + m) * 255.0).clamp(0.0, 255.0) as u8;
                let g_final = ((g_out + m) * 255.0).clamp(0.0, 255.0) as u8;
                let b_final = ((b_out + m) * 255.0).clamp(0.0, 255.0) as u8;

                result.put_pixel(x, y, Rgb([r_final, g_final, b_final]));
            }
        }

        result
    }

    /// Calculate tile positions to cover the entire image
    fn get_tile_positions(&self, img_width: u32, img_height: u32) -> Vec<(u32, u32)> {
        let mut positions = Vec::new();

        if img_width <= self.tile_size && img_height <= self.tile_size {
            positions.push((0, 0));
            return positions;
        }

        let stride = self.tile_size.saturating_sub(self.overlap);
        
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
    pub async fn process(&self, img: RgbImage) -> StageResult<PreprocessOutput> {
        let start = Instant::now();

        let original_width = img.width();
        let original_height = img.height();

        // Resize image to fit tile_size on longer dimension while preserving aspect ratio
        // This ensures the entire image fits in a single tile
        let resized_img = self.resize_to_fit(&img);
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
            let overlap = self.overlap;
            move || {
                let preprocess_stage = PreprocessStage::new(tile_size, overlap);
                preprocess_stage.extract_tiles(&deshadowed_img)
            }
        }).await.unwrap();
        
        let output = PreprocessOutput {
            tiles,
            original_width,
            original_height,
            resized_width,
            resized_height,
            resized_image: resized_img,  // Keep original resized for annotation
        };

        StageResult {
            data: output,
            stage_name: "Preprocess".to_string(),
            duration_ms: start.elapsed().as_secs_f32() * 1000.0,
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
    pub async fn process(&self, input: PreprocessOutput) -> StageResult<ExecutionOutput> {
        let start = Instant::now();

        // OPTIMIZATION: Pre-allocate detection vector (typical: 5-15 detections per tile)
        let mut all_detections = Vec::with_capacity(input.tiles.len() * 10);

        // Submit all tiles to batch executor
        let mut responses = Vec::new();
        let submit_start = Instant::now();
        
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
            let tile_start = Instant::now();
            let response_rx = self.frame_executor.detect_async(image_data);
            
            // Only add to responses if frame wasn't dropped due to backpressure
            if let Some(rx) = response_rx {
                responses.push((tile.tile_idx, tile.offset_x, tile.offset_y, rx, tile_start));
            } else {
                log::warn!("⚠️  Tile {} dropped due to detector backpressure", tile.tile_idx);
            }
        }
        let submit_time = submit_start.elapsed();

        // Collect results asynchronously and track per-tile timing
        let mut tile_results = Vec::new();
        for (tile_idx, offset_x, offset_y, response_rx, tile_start) in responses {
            // Use tokio's blocking recv since std::sync::mpsc::Receiver is not async
            match task::spawn_blocking(move || response_rx.recv()).await.unwrap() {
                Ok(Ok(detections)) => {
                    let tile_time = tile_start.elapsed();
                    tile_results.push((tile_idx, offset_x, offset_y, tile_time, detections));
                }
                _ => {
                    log::warn!("⚠️  Failed to receive detection result for tile {}", tile_idx);
                }
            }
        }

        // PARALLEL: Process all tile detections concurrently
        let processed_detections: Vec<(usize, std::time::Duration, Vec<TileDetection>)> = task::spawn_blocking({
            let tile_results = tile_results.clone();
            let tile_size = self.tile_size;
            let allowed_classes = self.allowed_classes.clone();
            move || {
                tile_results
                    .par_iter()
                    .map(|(tile_idx, offset_x, offset_y, tile_time, detections)| {
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

                        (*tile_idx, *tile_time, tile_dets)
                    })
                    .collect()
            }
        }).await.unwrap();

        // Collect all detections and timing info
        let mut tile_times = Vec::new();
        for (tile_idx, tile_time, tile_dets) in processed_detections {
            tile_times.push((tile_idx, tile_time, tile_dets.len()));
            all_detections.extend(tile_dets);
        }
        
        // Log per-tile performance
        if !tile_times.is_empty() {
            log::debug!("📊 Per-Tile End-to-End Latency (submission → response):");
            for (idx, time, det_count) in &tile_times {
                let ms = time.as_secs_f32() * 1000.0;
                let fps = 1000.0 / ms;
                log::debug!("  Tile #{}: {:.1}ms ({:.1} FPS) - {} detections", idx, ms, fps, det_count);
            }
            let total_ms: f32 = tile_times.iter().map(|(_, t, _)| t.as_secs_f32() * 1000.0).sum();
            let avg_ms = total_ms / tile_times.len() as f32;
            let avg_fps = 1000.0 / avg_ms;
            log::info!("  Frame latency: {:.1}ms ({:.1} FPS)", avg_ms, avg_fps);
            log::debug!("  Submit time: {:.1}ms", submit_time.as_secs_f32() * 1000.0);
        }

        let output = ExecutionOutput {
            detections: all_detections,
            original_width: input.original_width,
            original_height: input.original_height,
            resized_width: input.resized_width,
            resized_height: input.resized_height,
            resized_image: input.resized_image,
        };

        StageResult {
            data: output,
            stage_name: "Execution".to_string(),
            duration_ms: start.elapsed().as_secs_f32() * 1000.0,
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

    /// Calculate IoU between two bounding boxes
    /// OPTIMIZATION: Inline for performance + early exit for non-overlapping boxes
    #[inline]
    fn calculate_iou(a: &TileDetection, b: &TileDetection) -> f32 {
        // Early exit: Check if boxes don't overlap at all
        if a.x > b.x + b.w || b.x > a.x + a.w || a.y > b.y + b.h || b.y > a.y + a.h {
            return 0.0;
        }

        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.w).min(b.x + b.w);
        let y2 = (a.y + a.h).min(b.y + b.h);

        let intersection = (x2 - x1) * (y2 - y1);
        let area_a = a.w * a.h;
        let area_b = b.w * b.h;
        let union = area_a + area_b - intersection;

        intersection / union
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
    pub async fn process(&self, input: ExecutionOutput) -> StageResult<PostprocessOutput> {
        let start = Instant::now();

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

        let output = PostprocessOutput {
            detections: filtered_detections,
            original_width: input.original_width,
            original_height: input.original_height,
            resized_width: input.resized_width,
            resized_height: input.resized_height,
            resized_image: input.resized_image,
            duplicates_removed,
            nested_removed,
        };

        StageResult {
            data: output,
            stage_name: "Postprocess".to_string(),
            duration_ms: start.elapsed().as_secs_f32() * 1000.0,
        }
    }
}

/// Detection pipeline configuration
pub struct PipelineConfig {
    pub overlap: u32,
    pub allowed_classes: Vec<u32>,
    pub iou_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            overlap: 64,
            allowed_classes: vec![2, 3, 4, 7], // car, motorcycle, airplane, truck
            iou_threshold: 0.5,
        }
    }
}

/// Detection pipeline that processes a single image through three stages
/// The detector pool is shared across multiple pipeline executions
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

        let preprocess = PreprocessStage::new(tile_size, config.overlap);
        let execution = ExecutionStage::new(frame_executor, tile_size, config.allowed_classes);
        let postprocess = PostprocessStage::new(config.iou_threshold);

        Self {
            preprocess,
            execution,
            postprocess,
        }
    }

    /// Process image with detection completion callback (asynchronous)
    /// Each stage executes asynchronously when its input becomes available
    /// The callback is invoked immediately after detection completes, before returning results
    pub async fn process_with_callback<F>(&self, img: RgbImage, callback: F) -> Result<PostprocessOutput, String>
    where
        F: FnOnce(&[TileDetection], f32) + Send + 'static,
    {
        let start_time = Instant::now();
        
        // Create channels for stage coordination
        let (preprocess_tx, preprocess_rx) = oneshot::channel();
        let (execution_tx, execution_rx) = oneshot::channel();
        
        // Stage 1: Preprocess (tile extraction + shadow removal)
        let preprocess_stage = self.preprocess.clone();
        let preprocess_handle = tokio::spawn(async move {
            let result = preprocess_stage.process(img).await;
            let _ = preprocess_tx.send(result.data);
        });
        
        // Stage 2: Execution (batch detection via shared pool)
        let execution_stage = self.execution.clone();
        let execution_handle = tokio::spawn(async move {
            match preprocess_rx.await {
                Ok(preprocess_output) => {
                    let result = execution_stage.process(preprocess_output).await;
                    let _ = execution_tx.send(result.data);
                }
                Err(e) => {
                    log::error!("Failed to receive preprocess output: {}", e);
                }
            }
        });
        
        // Stage 3: Postprocess (NMS) - starts when execution completes
        let postprocess_stage = self.postprocess.clone();
        let postprocess_handle = tokio::spawn(async move {
            match execution_rx.await {
                Ok(execution_output) => {
                    let result = postprocess_stage.process(execution_output).await;
                    result.data
                }
                Err(e) => {
                    log::error!("Failed to receive execution output: {}", e);
                    // Return empty result on error
                    PostprocessOutput {
                        detections: Vec::new(),
                        original_width: 0,
                        original_height: 0,
                        resized_width: 0,
                        resized_height: 0,
                        resized_image: image::RgbImage::new(1, 1),
                        duplicates_removed: 0,
                        nested_removed: 0,
                    }
                }
            }
        });

        // Wait for all stages to complete
        let _ = preprocess_handle.await.map_err(|e| format!("Preprocess failed: {}", e))?;
        let _ = execution_handle.await.map_err(|e| format!("Execution failed: {}", e))?;
        let postprocess_output = postprocess_handle.await.map_err(|e| format!("Postprocess failed: {}", e))?;

        // Calculate processing time
        let processing_time = start_time.elapsed().as_secs_f32();
        
        // Invoke callback with detections and processing time
        callback(&postprocess_output.detections, processing_time);

        Ok(postprocess_output)
    }

    /// Process image asynchronously: Preprocess → Execution → Postprocess
    pub async fn process(&self, img: RgbImage) -> Result<PostprocessOutput, String> {
        // Stage 1: Preprocess (tile extraction + shadow removal)
        let preprocess_result = self.preprocess.process(img).await;

        // Stage 2: Execution (batch detection via shared pool)
        let execution_result = self.execution.process(preprocess_result.data).await;

        // Stage 3: Postprocess (NMS)
        let postprocess_result = self.postprocess.process(execution_result.data).await;

        Ok(postprocess_result.data)
    }

    /// Blocking wrapper for process_with_callback for compatibility with sync code
    /// This method blocks until the async processing completes
    pub fn process_with_callback_blocking<F>(&self, img: &RgbImage, callback: F) -> Result<PostprocessOutput, String>
    where
        F: FnOnce(&[TileDetection], f32),
    {
        let img_clone = img.clone();
        
        // Run the async pipeline synchronously
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.process(img_clone).await
            })
        });
        
        match result {
            Ok(output) => {
                // Calculate processing time (approximation since we don't have the exact timing)
                let processing_time = 0.0; // We could add timing here if needed
                callback(&output.detections, processing_time);
                Ok(output)
            }
            Err(e) => Err(e)
        }
    }

}
