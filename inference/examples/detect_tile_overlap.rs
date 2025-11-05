/// Tiled Detection with Overlapping Patches and Batch Processing
///
/// This example demonstrates detection on overlapping image tiles with efficient
/// batch processing for parallel GPU execution. The approach:
/// 1. Take original image (any size) - NO preprocessing
/// 2. Split into 640x640 overlapping tiles (64px overlap) covering the entire image
/// 3. Apply HSV-based shadow removal to each tile (minimalistic brightness enhancement)
/// 4. Submit tiles to worker pool -> batch executor collects them
/// 5. Batch executor processes multiple tiles in parallel (single GPU call with batch size)
/// 6. Merge detections with NMS to remove duplicates (postprocessing on whole image)
/// 7. Map coordinates back to original image space
///
/// Architecture: Workers -> Batch Executor (configurable batch_size & timeout_ms)
///
/// Usage:
///   cargo run --release --features metal --example detect_tile_overlap <image_path> [output_path]
use image::{Rgb, RgbImage};
use military_target_detector::batch_executor::BatchConfig;
use military_target_detector::detector_pool::DetectorPool;
use military_target_detector::image_utils::{draw_rect, draw_text, generate_class_color};
use military_target_detector::types::{DetectorConfig, ImageData, ImageFormat};
use std::env;
use std::time::Instant;

/// Apply simple HSV-based shadow removal (brightness enhancement)
/// Increases V channel for darker pixels (shadows) more than bright pixels
fn remove_shadows_hsv(img: &RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut result = RgbImage::new(width, height);

    // Convert RGB to HSV, enhance V channel, convert back
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
            // Increase brightness more for darker pixels
            let v_enhanced = if v < 0.5 {
                // Dark pixels: increase brightness significantly
                (v + 0.3).min(1.0)
            } else {
                // Bright pixels: slight increase
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

/// Configuration for tiled detection
struct TileConfig {
    /// Original image width
    img_width: u32,
    /// Original image height
    img_height: u32,
    /// Tile size (640x640 - YOLO native)
    tile_size: u32,
    /// Overlap between tiles (in pixels)
    overlap: u32,
}

impl TileConfig {
    fn new(img_width: u32, img_height: u32, overlap: u32) -> Self {
        Self {
            img_width,
            img_height,
            tile_size: 640, // YOLO's native resolution
            overlap,
        }
    }

    /// Calculate tile positions to cover the entire image
    fn get_tile_positions(&self) -> Vec<(u32, u32)> {
        let mut positions = Vec::new();

        // If image is smaller than tile size, just use one tile at (0,0)
        if self.img_width <= self.tile_size && self.img_height <= self.tile_size {
            positions.push((0, 0));
            return positions;
        }

        let stride = self.tile_size.saturating_sub(self.overlap);

        // Calculate number of tiles needed
        let tiles_x = if self.img_width <= self.tile_size {
            1
        } else {
            ((self.img_width - self.tile_size) as f32 / stride as f32).ceil() as u32 + 1
        };

        let tiles_y = if self.img_height <= self.tile_size {
            1
        } else {
            ((self.img_height - self.tile_size) as f32 / stride as f32).ceil() as u32 + 1
        };

        // Generate tile positions
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let x = if tx == tiles_x - 1 && self.img_width > self.tile_size {
                    // Last column - align to right edge
                    self.img_width.saturating_sub(self.tile_size)
                } else {
                    (tx * stride).min(self.img_width.saturating_sub(self.tile_size))
                };

                let y = if ty == tiles_y - 1 && self.img_height > self.tile_size {
                    // Last row - align to bottom edge
                    self.img_height.saturating_sub(self.tile_size)
                } else {
                    (ty * stride).min(self.img_height.saturating_sub(self.tile_size))
                };

                positions.push((x, y));
            }
        }

        // Remove duplicates (can happen when image dimension is close to tile_size)
        positions.sort();
        positions.dedup();

        positions
    }
}

/// Represents a tile with its position in the original image
struct Tile {
    image: RgbImage,
    offset_x: u32,
    offset_y: u32,
}

/// Extract overlapping tiles from the image
fn extract_tiles(img: &RgbImage, config: &TileConfig) -> Vec<Tile> {
    let positions = config.get_tile_positions();

    positions
        .iter()
        .map(|&(x, y)| {
            let crop_width = config.tile_size.min(img.width() - x);
            let crop_height = config.tile_size.min(img.height() - y);

            // Crop tile
            let cropped = image::imageops::crop_imm(img, x, y, crop_width, crop_height).to_image();

            // If tile is smaller than expected size, pad it to 640x640
            let mut tile = if crop_width < config.tile_size || crop_height < config.tile_size {
                let mut padded = RgbImage::new(config.tile_size, config.tile_size);
                // Fill with gray background
                for pixel in padded.pixels_mut() {
                    *pixel = Rgb([114, 114, 114]); // Gray padding (YOLO standard)
                }
                // Copy cropped image to top-left corner
                image::imageops::overlay(&mut padded, &cropped, 0, 0);
                padded
            } else {
                cropped
            };

            // Apply shadow removal to the tile
            tile = remove_shadows_hsv(&tile);

            Tile {
                image: tile,
                offset_x: x,
                offset_y: y,
            }
        })
        .collect()
}

/// Detection result with tile information
#[derive(Clone, Debug)]
struct TileDetection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    confidence: f32,
    class_id: i32,
    class_name: String,
    tile_idx: usize,
}

/// Calculate IoU (Intersection over Union) between two bounding boxes
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

/// Non-Maximum Suppression to remove duplicate detections
fn apply_nms(mut detections: Vec<TileDetection>, iou_threshold: f32) -> Vec<TileDetection> {
    // Sort by confidence (highest first)
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(detections[i].clone());

        // Suppress overlapping detections of the same class
        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }

            if detections[i].class_id == detections[j].class_id {
                let iou = calculate_iou(&detections[i], &detections[j]);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    keep
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🎯 YOLO Tiled Aerial Detector with Overlap - GPU Accelerated\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let image_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "test_data/yolo_airport.jpg".to_string()
    };

    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        "output_tiled_annotated.jpg".to_string()
    };

    let start_total = Instant::now();

    // Stage 1: Load original image (no resizing)
    print!("📷 Loading image... ");
    let original_img = image::open(&image_path)?.to_rgb8();
    let img_width = original_img.width();
    let img_height = original_img.height();
    println!("✓ ({}x{})", img_width, img_height);

    // Stage 2: Extract overlapping 640x640 tiles with shadow removal
    print!("🔲 Extracting overlapping tiles (with HSV shadow removal)... ");
    let start_tiles = Instant::now();
    let tile_config = TileConfig::new(img_width, img_height, 64); // 64px overlap
    let tiles = extract_tiles(&original_img, &tile_config);
    println!(
        "✓ ({} tiles, {}x{} YOLO native resolution, {:.1}ms)",
        tiles.len(),
        tile_config.tile_size,
        tile_config.tile_size,
        start_tiles.elapsed().as_secs_f32() * 1000.0
    );

    // Save tiles for inspection
    for (idx, tile) in tiles.iter().enumerate() {
        tile.image.save(format!("output_tile_{}.jpg", idx))?;
    }

    // Stage 3: Create detector pool with workers and batch executor
    print!("📦 Creating detector pool with batch executor on GPU... ");
    let start_load = Instant::now();

    let num_workers = 2; // Number of worker threads submitting commands
    let batch_config = BatchConfig {
        batch_size: 4,  // Process up to 4 tiles in parallel
        timeout_ms: 50, // Wait max 50ms for batch to fill
    };

    let detector_config = DetectorConfig {
        model_path: "../models/yolov8m_batch_fp16.onnx".to_string(),
        confidence_threshold: 0.22, // 22% confidence threshold
        nms_threshold: 0.45,
        input_size: (tile_config.tile_size, tile_config.tile_size), // 640x640
        use_gpu: true,
        ..Default::default()
    };

    let pool = DetectorPool::new(num_workers, detector_config, batch_config.clone())?;
    println!("✓ ({:.2}s)", start_load.elapsed().as_secs_f32());

    // Define allowed classes: car(2), motorcycle(3), airplane(4), truck(7)
    let allowed_classes: Vec<u32> = vec![2, 3, 4, 7];

    println!("\n⚙️  Tile Config:");
    println!(
        "  • Image size: {}x{}",
        tile_config.img_width, tile_config.img_height
    );
    println!(
        "  • Tile size: {}x{} (YOLO native)",
        tile_config.tile_size, tile_config.tile_size
    );
    println!("  • Overlap: {}px", tile_config.overlap);
    println!("  • Total tiles: {}", tiles.len());
    println!("  • Workers: {}", num_workers);
    println!("  • Batch size: {}", batch_config.batch_size);
    println!("  • Batch timeout: {}ms", batch_config.timeout_ms);

    // Stage 4: Run detection on all tiles using batch executor (parallel GPU execution)
    println!("\n🚀 Running detection on tiles (batch processing)...");
    let start_detect = Instant::now();

    let mut all_detections: Vec<TileDetection> = Vec::new();

    // Submit all tiles to worker pool and collect response receivers
    let mut responses = Vec::new();

    for (tile_idx, tile) in tiles.iter().enumerate() {
        // Convert tile to ImageData
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

        // Submit to pool (non-blocking)
        let response_rx = pool.detect_async(image_data);
        responses.push((tile_idx, response_rx, Instant::now()));
    }

    println!("  ✓ Submitted {} tiles to worker pool", tiles.len());

    // Collect results as they complete
    for (tile_idx, response_rx, tile_start) in responses {
        print!("  Tile {}/{}... ", tile_idx + 1, tiles.len());

        let detections = response_rx
            .recv()
            .map_err(|e| format!("Failed to receive: {}", e))?
            .map_err(|e| format!("Detection failed: {}", e))?;

        let tile_time = tile_start.elapsed().as_secs_f32();
        let tile_fps = 1.0 / tile_time;
        println!(
            "✓ {} detections ({:.0}ms, {:.1} FPS)",
            detections.len(),
            tile_time * 1000.0,
            tile_fps
        );

        // Transform coordinates from tile space to original image space
        // Filter by allowed classes
        let tile = &tiles[tile_idx];
        for det in detections {
            let class_id = det.class.id();

            // Skip if not in allowed classes
            if !allowed_classes.contains(&class_id) {
                continue;
            }

            // bbox is normalized (0-1), convert to pixels in tile space (640x640)
            let x_tile_px = det.bbox.x * tile_config.tile_size as f32;
            let y_tile_px = det.bbox.y * tile_config.tile_size as f32;
            let w_tile_px = det.bbox.width * tile_config.tile_size as f32;
            let h_tile_px = det.bbox.height * tile_config.tile_size as f32;

            // Add tile offset to get position in original image (no scaling needed)
            let x_final = x_tile_px + tile.offset_x as f32;
            let y_final = y_tile_px + tile.offset_y as f32;

            all_detections.push(TileDetection {
                x: x_final,
                y: y_final,
                w: w_tile_px,
                h: h_tile_px,
                confidence: det.confidence,
                class_id: det.class.id() as i32,
                class_name: format!("{:?}", det.class),
                tile_idx,
            });
        }
    }

    let detect_time = start_detect.elapsed();
    println!(
        "\n✓ Total detection time: {:.1}ms ({:.1} FPS across all tiles)",
        detect_time.as_secs_f32() * 1000.0,
        tiles.len() as f32 / detect_time.as_secs_f32()
    );

    // Stage 7: Apply NMS to remove duplicates
    print!("\n🔄 Applying NMS to remove duplicates... ");
    let before_nms = all_detections.len();
    let merged_detections = apply_nms(all_detections, 0.5);
    println!(
        "✓ ({} → {} detections)",
        before_nms,
        merged_detections.len()
    );

    // Stage 8: Print results
    println!("\n📊 Final Results:");
    println!("  • Total detections (before NMS): {}", before_nms);
    println!(
        "  • Final detections (after NMS): {}",
        merged_detections.len()
    );
    println!(
        "  • Duplicates removed: {}",
        before_nms - merged_detections.len()
    );

    if !merged_detections.is_empty() {
        println!("\n🎯 Detected Targets:\n");
        for (i, det) in merged_detections.iter().enumerate() {
            println!("  Detection #{}:", i + 1);
            println!("    Class: {} (ID: {})", det.class_name, det.class_id);
            println!("    Confidence: {:.1}%", det.confidence * 100.0);
            println!(
                "    Box: x={:.3}, y={:.3}, w={:.3}, h={:.3}",
                det.x / img_width as f32,
                det.y / img_height as f32,
                det.w / img_width as f32,
                det.h / img_height as f32
            );
            println!("    Tile: {}", det.tile_idx);
            println!();
        }
    }

    // Stage 8: Draw bounding boxes on the original image
    print!("🎨 Drawing bounding boxes... \n");
    let mut annotated_img = original_img.clone();

    for (i, det) in merged_detections.iter().enumerate() {
        // Generate color based on class ID (different colors for different classes)
        let color = generate_class_color(det.class_id as u32);

        // Coordinates are already in pixels for the original image
        // Draw rectangle
        draw_rect(
            &mut annotated_img,
            det.x as i32,
            det.y as i32,
            det.w as u32,
            det.h as u32,
            color,
            2,
        );

        // Draw label with white text and transparent background
        let label = format!("{} {:.0}%", det.class_name, det.confidence * 100.0);

        // Calculate label position with padding
        let padding = 3;
        let text_x = (det.x as i32 + padding).max(0);
        let text_y = (det.y as i32 + padding).max(0);

        // Only draw label if there's enough space
        if det.h > 20.0 && det.w > 30.0 {
            let text_color = Rgb([255, 255, 255]); // White text
            draw_text(
                &mut annotated_img,
                &label,
                text_x,
                text_y,
                text_color,
                None, // Transparent background
            );
        }

        println!(
            "  ✓ Drew box #{}: {} at ({:.0}, {:.0}) size {:.0}x{:.0}",
            i + 1,
            det.class_name,
            det.x,
            det.y,
            det.w,
            det.h
        );
    }

    println!("\n✓ Drawing complete");

    // Save annotated image
    annotated_img.save(&output_path)?;
    println!("💾 Saved annotated image to: {}", output_path);

    println!(
        "\n⏱️  Total time: {:.2}s",
        start_total.elapsed().as_secs_f32()
    );

    Ok(())
}
