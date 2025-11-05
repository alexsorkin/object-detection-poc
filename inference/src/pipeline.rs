/// Detection pipeline with pre-processing, execution, and post-processing stages
use crate::detector_pool::DetectorPool;
use crate::types::{ImageData, ImageFormat};
use image::{Rgb, RgbImage};
use std::sync::Arc;
use std::time::Instant;

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
    pub tiles: Vec<Tile>,
    pub original_width: u32,
    pub original_height: u32,
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
}

/// Execution output: detections from all tiles
#[derive(Clone)]
pub struct ExecutionOutput {
    pub detections: Vec<TileDetection>,
    pub original_width: u32,
    pub original_height: u32,
}

/// Final output after post-processing
#[derive(Clone)]
pub struct PostprocessOutput {
    pub detections: Vec<TileDetection>,
    pub original_width: u32,
    pub original_height: u32,
    pub duplicates_removed: usize,
    pub nested_removed: usize,
}

/// Pre-processing stage: Extract tiles and apply shadow removal
pub struct PreprocessStage {
    tile_size: u32,
    overlap: u32,
}

impl PreprocessStage {
    pub fn new(tile_size: u32, overlap: u32) -> Self {
        Self { tile_size, overlap }
    }

    /// Apply HSV-based shadow removal (brightness enhancement)
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

        let tiles_x = if img_width <= self.tile_size {
            1
        } else {
            ((img_width - self.tile_size) as f32 / stride as f32).ceil() as u32 + 1
        };

        let tiles_y = if img_height <= self.tile_size {
            1
        } else {
            ((img_height - self.tile_size) as f32 / stride as f32).ceil() as u32 + 1
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

    /// Extract tiles from image with shadow removal
    pub fn extract_tiles(&self, img: &RgbImage) -> Vec<Tile> {
        let positions = self.get_tile_positions(img.width(), img.height());

        positions
            .iter()
            .enumerate()
            .map(|(idx, &(x, y))| {
                let crop_width = self.tile_size.min(img.width() - x);
                let crop_height = self.tile_size.min(img.height() - y);

                let cropped =
                    image::imageops::crop_imm(img, x, y, crop_width, crop_height).to_image();

                // Pad if necessary
                let mut tile = if crop_width < self.tile_size || crop_height < self.tile_size {
                    let mut padded = RgbImage::new(self.tile_size, self.tile_size);
                    for pixel in padded.pixels_mut() {
                        *pixel = Rgb([114, 114, 114]);
                    }
                    image::imageops::overlay(&mut padded, &cropped, 0, 0);
                    padded
                } else {
                    cropped
                };

                // Apply shadow removal
                tile = self.remove_shadows_hsv(&tile);

                Tile {
                    image: tile,
                    offset_x: x,
                    offset_y: y,
                    tile_idx: idx,
                }
            })
            .collect()
    }

    /// Run pre-processing stage (synchronous)
    pub fn process(&self, img: &RgbImage) -> StageResult<PreprocessOutput> {
        let start = Instant::now();

        let tiles = self.extract_tiles(img);
        let output = PreprocessOutput {
            tiles,
            original_width: img.width(),
            original_height: img.height(),
        };

        StageResult {
            data: output,
            stage_name: "Preprocess".to_string(),
            duration_ms: start.elapsed().as_secs_f32() * 1000.0,
        }
    }
}

/// Execution stage: Run detection on tiles using shared detector pool
pub struct ExecutionStage {
    detector_pool: Arc<DetectorPool>,
    tile_size: u32,
    allowed_classes: Vec<u32>,
}

impl ExecutionStage {
    pub fn new(
        detector_pool: Arc<DetectorPool>,
        tile_size: u32,
        allowed_classes: Vec<u32>,
    ) -> Self {
        Self {
            detector_pool,
            tile_size,
            allowed_classes,
        }
    }

    /// Run detection on tiles (synchronous)
    pub fn process(&self, input: PreprocessOutput) -> StageResult<ExecutionOutput> {
        let start = Instant::now();

        let mut all_detections = Vec::new();

        // Submit all tiles to shared detector pool
        let mut responses = Vec::new();
        for tile in &input.tiles {
            let mut tile_data = Vec::new();
            for pixel in tile.image.pixels() {
                tile_data.push(pixel[0]);
                tile_data.push(pixel[1]);
                tile_data.push(pixel[2]);
            }

            let image_data = ImageData::new(
                tile_data,
                tile.image.width(),
                tile.image.height(),
                ImageFormat::RGB,
            );

            let response_rx = self.detector_pool.detect_async(image_data);
            responses.push((tile.tile_idx, tile.offset_x, tile.offset_y, response_rx));
        }

        // Collect results
        for (tile_idx, offset_x, offset_y, response_rx) in responses {
            if let Ok(Ok(detections)) = response_rx.recv() {
                for det in detections {
                    let class_id = det.class.id();

                    if !self.allowed_classes.contains(&class_id) {
                        continue;
                    }

                    // Convert normalized coordinates to pixels in tile space
                    let x_tile_px = det.bbox.x * self.tile_size as f32;
                    let y_tile_px = det.bbox.y * self.tile_size as f32;
                    let w_tile_px = det.bbox.width * self.tile_size as f32;
                    let h_tile_px = det.bbox.height * self.tile_size as f32;

                    // Add tile offset to get position in original image
                    let x_final = x_tile_px + offset_x as f32;
                    let y_final = y_tile_px + offset_y as f32;

                    all_detections.push(TileDetection {
                        x: x_final,
                        y: y_final,
                        w: w_tile_px,
                        h: h_tile_px,
                        confidence: det.confidence,
                        class_id,
                        class_name: format!("{:?}", det.class),
                        tile_idx,
                    });
                }
            }
        }

        let output = ExecutionOutput {
            detections: all_detections,
            original_width: input.original_width,
            original_height: input.original_height,
        };

        StageResult {
            data: output,
            stage_name: "Execution".to_string(),
            duration_ms: start.elapsed().as_secs_f32() * 1000.0,
        }
    }
}

/// Post-processing stage: Apply NMS and prepare final results
pub struct PostprocessStage {
    iou_threshold: f32,
}

impl PostprocessStage {
    pub fn new(iou_threshold: f32) -> Self {
        Self { iou_threshold }
    }

    /// Calculate IoU between two bounding boxes
    fn calculate_iou(a: &TileDetection, b: &TileDetection) -> f32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.w).min(b.x + b.w);
        let y2 = (a.y + a.h).min(b.y + b.h);

        if x2 < x1 || y2 < y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area_a = a.w * a.h;
        let area_b = b.w * b.h;
        let union = area_a + area_b - intersection;

        intersection / union
    }

    /// Check if detection `inner` is fully contained within detection `outer`
    fn is_fully_contained(inner: &TileDetection, outer: &TileDetection) -> bool {
        let inner_x1 = inner.x;
        let inner_y1 = inner.y;
        let inner_x2 = inner.x + inner.w;
        let inner_y2 = inner.y + inner.h;

        let outer_x1 = outer.x;
        let outer_y1 = outer.y;
        let outer_x2 = outer.x + outer.w;
        let outer_y2 = outer.y + outer.h;

        inner_x1 >= outer_x1 && inner_y1 >= outer_y1 && inner_x2 <= outer_x2 && inner_y2 <= outer_y2
    }

    /// Filter out detections that are fully contained within other detections
    fn filter_nested_detections(&self, detections: Vec<TileDetection>) -> Vec<TileDetection> {
        let mut filtered = Vec::new();

        for (i, det) in detections.iter().enumerate() {
            let mut is_contained = false;

            // Check if this detection is contained in any other detection
            for (j, other) in detections.iter().enumerate() {
                if i != j && Self::is_fully_contained(det, other) {
                    is_contained = true;
                    break;
                }
            }

            if !is_contained {
                filtered.push(det.clone());
            }
        }

        filtered
    }

    /// Apply NMS to remove duplicates
    fn apply_nms(&self, mut detections: Vec<TileDetection>) -> Vec<TileDetection> {
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }

            keep.push(detections[i].clone());

            for j in (i + 1)..detections.len() {
                if suppressed[j] {
                    continue;
                }

                if detections[i].class_id == detections[j].class_id {
                    let iou = Self::calculate_iou(&detections[i], &detections[j]);
                    if iou > self.iou_threshold {
                        suppressed[j] = true;
                    }
                }
            }
        }

        keep
    }

    /// Run post-processing stage (synchronous)
    pub fn process(&self, input: ExecutionOutput) -> StageResult<PostprocessOutput> {
        let start = Instant::now();

        let before_nms = input.detections.len();

        // Step 1: Apply NMS to remove duplicates based on IoU
        let after_nms = self.apply_nms(input.detections);
        let duplicates_removed = before_nms - after_nms.len();

        // Step 2: Filter nested detections (fully contained boxes)
        let before_nested_filter = after_nms.len();
        let filtered_detections = self.filter_nested_detections(after_nms);
        let nested_removed = before_nested_filter - filtered_detections.len();

        let output = PostprocessOutput {
            detections: filtered_detections,
            original_width: input.original_width,
            original_height: input.original_height,
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
    pub tile_size: u32,
    pub overlap: u32,
    pub allowed_classes: Vec<u32>,
    pub iou_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tile_size: 640,
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
    /// Create a new detection pipeline with shared detector pool
    pub fn new(detector_pool: Arc<DetectorPool>, config: PipelineConfig) -> Self {
        let preprocess = PreprocessStage::new(config.tile_size, config.overlap);
        let execution =
            ExecutionStage::new(detector_pool, config.tile_size, config.allowed_classes);
        let postprocess = PostprocessStage::new(config.iou_threshold);

        Self {
            preprocess,
            execution,
            postprocess,
        }
    }

    /// Process a single image through the entire pipeline (synchronous)
    /// Stages execute sequentially: Preprocess → Execution → Postprocess
    pub fn process(&self, img: &RgbImage) -> Result<PostprocessOutput, String> {
        // Stage 1: Preprocess (tile extraction + shadow removal)
        let preprocess_result = self.preprocess.process(img);

        // Stage 2: Execution (batch detection via shared pool)
        let execution_result = self.execution.process(preprocess_result.data);

        // Stage 3: Postprocess (NMS)
        let postprocess_result = self.postprocess.process(execution_result.data);

        Ok(postprocess_result.data)
    }

    /// Process image with detailed timing for each stage
    pub fn process_with_timing(
        &self,
        img: &RgbImage,
    ) -> Result<(PostprocessOutput, PipelineTiming), String> {
        let start_total = Instant::now();

        // Stage 1: Preprocess
        let preprocess_result = self.preprocess.process(img);
        let preprocess_time = preprocess_result.duration_ms;

        // Stage 2: Execution
        let execution_result = self.execution.process(preprocess_result.data);
        let execution_time = execution_result.duration_ms;

        // Stage 3: Postprocess
        let postprocess_result = self.postprocess.process(execution_result.data);
        let postprocess_time = postprocess_result.duration_ms;

        let total_time = start_total.elapsed().as_secs_f32() * 1000.0;

        log::debug!(
            "Pipeline timing - Preprocess: {:.1}ms, Execution: {:.1}ms, Postprocess: {:.1}ms, Total: {:.1}ms",
            preprocess_time, execution_time, postprocess_time, total_time
        );

        let timing = PipelineTiming {
            preprocess_ms: preprocess_time,
            execution_ms: execution_time,
            postprocess_ms: postprocess_time,
            total_ms: total_time,
        };

        Ok((postprocess_result.data, timing))
    }
}
