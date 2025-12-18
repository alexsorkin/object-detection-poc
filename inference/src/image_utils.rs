/// Image utilities for drawing and labeling detections
use image::{Rgb, RgbImage};
use opencv::{
    core::{Mat, Size, Vector},
    imgproc,
    prelude::*,
};

// Import for TileDetection trait implementation
use crate::frame_pipeline::TileDetection;

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for shadow removal
/// Uses OpenCV's CLAHE on the L channel of LAB color space
/// Works directly on OpenCV Mat in BGR format (OpenCV's native format)
///
/// # Arguments
/// * `bgr_mat` - The input BGR Mat (OpenCV's native format from video capture)
///
/// # Returns
/// Result containing the enhanced BGR Mat or an error
pub fn remove_shadows_clahe(bgr_mat: &Mat) -> Result<Mat, Box<dyn std::error::Error>> {
    // Convert BGR to LAB color space
    let mut lab_mat = Mat::default();

    // macOS OpenCV requires AlgorithmHint parameter, Linux does not
    #[cfg(target_os = "macos")]
    imgproc::cvt_color(
        bgr_mat,
        &mut lab_mat,
        imgproc::COLOR_BGR2Lab,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    #[cfg(not(target_os = "macos"))]
    imgproc::cvt_color(bgr_mat, &mut lab_mat, imgproc::COLOR_BGR2Lab, 0)?;

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

    // Convert back to BGR
    let mut bgr_enhanced = Mat::default();

    // macOS OpenCV requires AlgorithmHint parameter, Linux does not
    #[cfg(target_os = "macos")]
    imgproc::cvt_color(
        &lab_enhanced,
        &mut bgr_enhanced,
        imgproc::COLOR_Lab2BGR,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    #[cfg(not(target_os = "macos"))]
    imgproc::cvt_color(&lab_enhanced, &mut bgr_enhanced, imgproc::COLOR_Lab2BGR, 0)?;

    Ok(bgr_enhanced)
}

/// Calculate scale factors for resizing image to target size while preserving aspect ratio
///
/// # Arguments
/// * `width` - Original image width
/// * `height` - Original image height
/// * `target_size` - Target size for the largest dimension
///
/// # Returns
/// A tuple (scale_x, scale_y) representing the scale factors for x and y dimensions
pub fn calculate_scale_factors(width: u32, height: u32, target_size: f32) -> (f32, f32) {
    let scale = if width > height {
        (target_size / width as f32).min(1.0)
    } else {
        (target_size / height as f32).min(1.0)
    };

    let resized_width = (width as f32 * scale) as u32;
    let resized_height = (height as f32 * scale) as u32;

    (
        width as f32 / resized_width as f32,
        height as f32 / resized_height as f32,
    )
}

/// Prepare annotation data for detection results in parallel
///
/// # Arguments
/// * `predictions` - Vector of detection predictions  
/// * `scale_x` - X scale factor for coordinate transformation
/// * `scale_y` - Y scale factor for coordinate transformation
///
/// # Returns
/// Vector of annotation data tuples (scaled_x, scaled_y, scaled_w, scaled_h, color, label, show_label)
pub fn prepare_annotation_data<T>(
    predictions: &[T],
    scale_x: f32,
    scale_y: f32,
) -> Vec<(i32, i32, u32, u32, Rgb<u8>, String, bool)>
where
    T: DetectionData + Sync,
{
    use rayon::prelude::*;

    predictions
        .par_iter()
        .filter_map(|det| {
            let color = generate_class_color(det.get_class_id());
            let scaled_x = (det.get_x() * scale_x) as i32;
            let scaled_y = (det.get_y() * scale_y) as i32;
            let scaled_w = det.get_w() * scale_x;
            let scaled_h = det.get_h() * scale_y;

            // Validate dimensions before casting to u32
            if scaled_w <= 0.0 || scaled_h <= 0.0 {
                log::warn!(
                    "Invalid bbox dimensions - width: {}, height: {}, skipping",
                    scaled_w,
                    scaled_h
                );
                return None;
            }

            let scaled_w = scaled_w.max(1.0) as u32;
            let scaled_h = scaled_h.max(1.0) as u32;

            let label = if let Some(track_id) = det.get_track_id() {
                format!(
                    "#{} {} {:.0}%",
                    track_id,
                    det.get_class_name(),
                    det.get_confidence() * 100.0
                )
            } else {
                format!(
                    "{} {:.0}%",
                    det.get_class_name(),
                    det.get_confidence() * 100.0
                )
            };

            let show_label = scaled_h > 20 && scaled_w > 30;

            Some((
                scaled_x, scaled_y, scaled_w, scaled_h, color, label, show_label,
            ))
        })
        .collect()
}

/// Trait for accessing detection data in a generic way
pub trait DetectionData {
    fn get_class_id(&self) -> u32;
    fn get_x(&self) -> f32;
    fn get_y(&self) -> f32;
    fn get_w(&self) -> f32;
    fn get_h(&self) -> f32;
    fn get_track_id(&self) -> Option<u32>;
    fn get_class_name(&self) -> &str;
    fn get_confidence(&self) -> f32;
}

/// Implementation of DetectionData trait for TileDetection
impl DetectionData for TileDetection {
    fn get_class_id(&self) -> u32 {
        self.class_id
    }

    fn get_x(&self) -> f32 {
        self.x
    }

    fn get_y(&self) -> f32 {
        self.y
    }

    fn get_w(&self) -> f32 {
        self.w
    }

    fn get_h(&self) -> f32 {
        self.h
    }

    fn get_track_id(&self) -> Option<u32> {
        self.track_id
    }

    fn get_class_name(&self) -> &str {
        &self.class_name
    }

    fn get_confidence(&self) -> f32 {
        self.confidence
    }
}

/// Get 5x7 bitmap pattern for a character
fn get_char_pattern(ch: char) -> [u8; 7] {
    match ch {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        '%' => [
            0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
        ],
        ' ' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        '_' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
        ],
        '#' => [
            0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010,
        ],
        _ => [
            0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
        ], // Box for unknown chars
    }
}

/// Draw text on image using 5x7 bitmap font
///
/// # Arguments
/// * `img` - The image to draw on
/// * `text` - The text to draw
/// * `x` - X coordinate (top-left)
/// * `y` - Y coordinate (top-left)
/// * `color` - Text color
/// * `bg_color` - Optional background color
pub fn draw_text(
    img: &mut RgbImage,
    text: &str,
    x: i32,
    y: i32,
    color: Rgb<u8>,
    bg_color: Option<Rgb<u8>>,
) {
    let char_width = 5; // 5 pixels (no extra space for smaller font)
    let char_height = 7;

    // Draw background if provided
    if let Some(bg) = bg_color {
        let text_width = (text.len() as i32 * char_width) + 2;
        let text_height = char_height + 2;

        for dy in 0..text_height {
            for dx in 0..text_width {
                let px = x + dx;
                let py = y + dy;
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(px as u32, py as u32, bg);
                }
            }
        }
    }

    // Draw each character using bitmap patterns
    for (i, ch) in text.to_uppercase().chars().enumerate() {
        let char_x = x + 1 + (i as i32 * char_width);
        let char_y = y + 1;

        let pattern = get_char_pattern(ch);

        // Draw the character bitmap
        for (row, &bits) in pattern.iter().enumerate() {
            for col in 0..5 {
                // Check if this bit is set (reading from left to right)
                if (bits >> (4 - col)) & 1 == 1 {
                    let px = char_x + col;
                    let py = char_y + row as i32;
                    if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height()
                    {
                        img.put_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }
}

/// Generate a deterministic color based on class ID
/// Uses a hash-like approach to generate colors that are darker than typical backgrounds
///
/// # Arguments
/// * `class_id` - The class ID to generate a color for
///
/// # Returns
/// An RGB color that is deterministic per class ID
pub fn generate_class_color(class_id: u32) -> Rgb<u8> {
    // Use a simple hash to generate pseudo-random but deterministic values
    let mut hash = class_id.wrapping_mul(2654435761); // Golden ratio prime

    // Generate RGB components
    let r = ((hash & 0xFF) as u8) as u16;
    hash = hash.wrapping_mul(2654435761);
    let g = ((hash & 0xFF) as u8) as u16;
    hash = hash.wrapping_mul(2654435761);
    let b = ((hash & 0xFF) as u8) as u16;

    // Ensure colors are reasonably dark (max 180 per channel)
    // and have good saturation (at least one channel > 100)
    let max_value = 180u16;
    let min_bright = 100u16;

    let r = (r.min(max_value)).max(if r > g && r > b { min_bright } else { 40 });
    let g = (g.min(max_value)).max(if g > r && g > b { min_bright } else { 40 });
    let b = (b.min(max_value)).max(if b > r && b > g { min_bright } else { 40 });

    Rgb([r as u8, g as u8, b as u8])
}

/// Resize image with aspect ratio preserved (letterboxing)
/// Creates a standardized input size with gray padding
///
/// # Arguments
/// * `img` - The image to resize
/// * `target_size` - Target size (width and height, image will be square)
///
/// # Returns
/// A new image resized to target_size x target_size with aspect ratio preserved
pub fn resize_with_letterbox(img: &RgbImage, target_size: u32) -> RgbImage {
    // Create gray canvas
    let mut canvas = RgbImage::from_pixel(target_size, target_size, Rgb([128, 128, 128]));

    // Calculate scaling to fit within target size
    let scale =
        (target_size as f32 / img.width() as f32).min(target_size as f32 / img.height() as f32);
    let new_width = (img.width() as f32 * scale) as u32;
    let new_height = (img.height() as f32 * scale) as u32;

    // Resize maintaining aspect ratio
    let resized = image::imageops::resize(
        img,
        new_width,
        new_height,
        image::imageops::FilterType::Lanczos3,
    );

    // Calculate centering offsets
    let x_offset = (target_size - new_width) / 2;
    let y_offset = (target_size - new_height) / 2;

    // Paste resized image onto canvas (letterbox with gray)
    image::imageops::overlay(&mut canvas, &resized, x_offset as i64, y_offset as i64);

    canvas
}

/// Draw a thick rectangle on an image
///
/// # Arguments
/// * `img` - The image to draw on
/// * `x` - X coordinate of top-left corner
/// * `y` - Y coordinate of top-left corner
/// * `width` - Width of the rectangle
/// * `height` - Height of the rectangle
/// * `color` - Color of the rectangle
/// * `thickness` - Thickness of the border (in pixels)
pub fn draw_rect(
    img: &mut RgbImage,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    color: Rgb<u8>,
    thickness: i32,
) {
    use imageproc::drawing::draw_hollow_rect_mut;
    use imageproc::rect::Rect;

    let rect = Rect::at(x, y).of_size(width, height);

    // Draw thick border by drawing multiple rectangles
    for offset in 0..thickness {
        let expanded_rect = Rect::at(rect.left() - offset, rect.top() - offset).of_size(
            rect.width() + (offset * 2) as u32,
            rect.height() + (offset * 2) as u32,
        );
        draw_hollow_rect_mut(img, expanded_rect, color);
    }
}

/// Parallel batch text rendering - draw multiple text labels in parallel
///
/// # Arguments
/// * `img` - The image to draw on
/// * `labels` - Vector of (text, x, y, color, bg_color) tuples
pub fn draw_text_batch(img: &mut RgbImage, labels: &[(&str, i32, i32, Rgb<u8>, Option<Rgb<u8>>)]) {
    use rayon::prelude::*;

    // Collect all pixel modifications in parallel
    let pixel_updates: Vec<Vec<(u32, u32, Rgb<u8>)>> = labels
        .par_iter()
        .map(|(text, x, y, color, bg_color)| {
            let mut pixels = Vec::new();
            let char_width = 5;
            let char_height = 7;

            // Collect background pixels
            if let Some(bg) = bg_color {
                let text_width = (text.len() as i32 * char_width) + 2;
                let text_height = char_height + 2;

                for dy in 0..text_height {
                    for dx in 0..text_width {
                        let px = x + dx;
                        let py = y + dy;
                        if px >= 0
                            && py >= 0
                            && (px as u32) < img.width()
                            && (py as u32) < img.height()
                        {
                            pixels.push((px as u32, py as u32, *bg));
                        }
                    }
                }
            }

            // Collect character pixels
            for (i, ch) in text.to_uppercase().chars().enumerate() {
                let char_x = x + 1 + (i as i32 * char_width);
                let char_y = y + 1;
                let pattern = get_char_pattern(ch);

                for (row, &bits) in pattern.iter().enumerate() {
                    for col in 0..5 {
                        if (bits >> (4 - col)) & 1 == 1 {
                            let px = char_x + col;
                            let py = char_y + row as i32;
                            if px >= 0
                                && py >= 0
                                && (px as u32) < img.width()
                                && (py as u32) < img.height()
                            {
                                pixels.push((px as u32, py as u32, *color));
                            }
                        }
                    }
                }
            }

            pixels
        })
        .collect();

    // Apply all pixel updates sequentially (fast since it's just memory writes)
    for pixel_set in pixel_updates {
        for (x, y, color) in pixel_set {
            img.put_pixel(x, y, color);
        }
    }
}

/// Parallel batch rectangle drawing - draw multiple rectangles in parallel
///
/// # Arguments
/// * `img` - The image to draw on
/// * `rects` - Vector of (x, y, width, height, color, thickness) tuples
pub fn draw_rect_batch(img: &mut RgbImage, rects: &[(i32, i32, u32, u32, Rgb<u8>, i32)]) {
    use imageproc::rect::Rect;
    use rayon::prelude::*;

    // Collect all pixels for all rectangles in parallel
    let pixel_updates: Vec<Vec<(u32, u32, Rgb<u8>)>> = rects
        .par_iter()
        .filter_map(|(x, y, width, height, color, thickness)| {
            // Validate dimensions before creating Rect
            if *width == 0 || *height == 0 {
                eprintln!(
                    "Warning: Cannot draw rect with zero dimensions - width: {}, height: {}",
                    width, height
                );
                return None;
            }

            let mut pixels = Vec::new();
            let rect = Rect::at(*x, *y).of_size(*width, *height);

            // Collect pixels for all thickness layers
            for offset in 0..*thickness {
                let expanded_rect = Rect::at(rect.left() - offset, rect.top() - offset).of_size(
                    rect.width() + (offset * 2) as u32,
                    rect.height() + (offset * 2) as u32,
                );

                // Collect hollow rectangle pixels
                // Top edge
                for dx in 0..expanded_rect.width() {
                    let px = expanded_rect.left() + dx as i32;
                    let py = expanded_rect.top();
                    if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height()
                    {
                        pixels.push((px as u32, py as u32, *color));
                    }
                }
                // Bottom edge
                for dx in 0..expanded_rect.width() {
                    let px = expanded_rect.left() + dx as i32;
                    let py = expanded_rect.bottom() - 1;
                    if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height()
                    {
                        pixels.push((px as u32, py as u32, *color));
                    }
                }
                // Left edge
                for dy in 0..expanded_rect.height() {
                    let px = expanded_rect.left();
                    let py = expanded_rect.top() + dy as i32;
                    if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height()
                    {
                        pixels.push((px as u32, py as u32, *color));
                    }
                }
                // Right edge
                for dy in 0..expanded_rect.height() {
                    let px = expanded_rect.right() - 1;
                    let py = expanded_rect.top() + dy as i32;
                    if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height()
                    {
                        pixels.push((px as u32, py as u32, *color));
                    }
                }
            }

            Some(pixels)
        })
        .collect();

    // Apply all pixel updates sequentially
    for pixel_set in pixel_updates {
        for (x, y, color) in pixel_set {
            img.put_pixel(x, y, color);
        }
    }
}
