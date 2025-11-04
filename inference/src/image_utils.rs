/// Image utilities for drawing and labeling detections
use image::{Rgb, RgbImage};

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
