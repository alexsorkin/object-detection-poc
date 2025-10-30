//! Image preprocessing utilities for military target detection

use crate::error::{DetectionError, Result};
use crate::types::{ImageData, ImageFormat};
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array4, Axis};
use std::path::Path;

/// Image preprocessor for preparing input data for the model
pub struct ImagePreprocessor {
    /// Target input size (width, height)
    input_size: (u32, u32),
    /// Whether to normalize pixel values to [0, 1]
    normalize: bool,
    /// Mean values for normalization (RGB)
    mean: [f32; 3],
    /// Standard deviation values for normalization (RGB)
    std: [f32; 3],
}

impl ImagePreprocessor {
    /// Create new image preprocessor
    pub fn new(input_size: (u32, u32)) -> Self {
        Self {
            input_size,
            normalize: true,
            // ImageNet mean and std values
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Set normalization parameters
    pub fn with_normalization(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = mean;
        self.std = std;
        self
    }

    /// Disable normalization
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Load image from file path
    pub fn load_image<P: AsRef<Path>>(&self, path: P) -> Result<ImageData> {
        let img = image::open(path)
            .map_err(|e| DetectionError::preprocessing(format!("Failed to load image: {}", e)))?;

        self.from_dynamic_image(img)
    }

    /// Convert from DynamicImage
    pub fn from_dynamic_image(&self, img: DynamicImage) -> Result<ImageData> {
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();

        Ok(ImageData::new(
            rgb_img.into_raw(),
            width,
            height,
            ImageFormat::RGB,
        ))
    }

    /// Convert raw image data to input tensor
    pub fn preprocess(&self, image: &ImageData) -> Result<Array4<f32>> {
        // Validate input
        if !image.validate() {
            return Err(DetectionError::preprocessing(
                "Invalid image data: size mismatch".to_string(),
            ));
        }

        // Convert to RGB if needed
        let rgb_data = self.convert_to_rgb(image)?;

        // Resize image to model input size
        let resized_data = self.resize_image(&rgb_data, image.width, image.height)?;

        // Convert to tensor format: [batch_size, channels, height, width]
        let tensor = self.to_tensor(&resized_data)?;

        Ok(tensor)
    }

    /// Process multiple images in batch
    pub fn preprocess_batch(&self, images: &[ImageData]) -> Result<Array4<f32>> {
        if images.is_empty() {
            return Err(DetectionError::preprocessing(
                "Empty image batch".to_string(),
            ));
        }

        let mut tensors = Vec::new();
        for image in images {
            let tensor = self.preprocess(image)?;
            tensors.push(tensor.remove_axis(Axis(0))); // Remove batch dimension
        }

        // Stack tensors along batch dimension
        let tensor_views: Vec<_> = tensors.iter().map(|t| t.view()).collect();
        let batch_tensor = ndarray::stack(Axis(0), &tensor_views).map_err(|e| {
            DetectionError::preprocessing(format!("Failed to stack tensors: {}", e))
        })?;

        Ok(batch_tensor)
    }

    /// Convert image to RGB format
    fn convert_to_rgb(&self, image: &ImageData) -> Result<Vec<u8>> {
        match image.format {
            ImageFormat::RGB => Ok(image.data.clone()),
            ImageFormat::BGR => {
                // Convert BGR to RGB by swapping R and B channels
                let mut rgb_data = Vec::with_capacity(image.data.len());
                for chunk in image.data.chunks_exact(3) {
                    rgb_data.push(chunk[2]); // R (was B)
                    rgb_data.push(chunk[1]); // G
                    rgb_data.push(chunk[0]); // B (was R)
                }
                Ok(rgb_data)
            }
            ImageFormat::RGBA => {
                // Drop alpha channel
                let mut rgb_data = Vec::with_capacity(image.data.len() * 3 / 4);
                for chunk in image.data.chunks_exact(4) {
                    rgb_data.push(chunk[0]); // R
                    rgb_data.push(chunk[1]); // G
                    rgb_data.push(chunk[2]); // B
                }
                Ok(rgb_data)
            }
            ImageFormat::BGRA => {
                // Convert BGRA to RGB (drop alpha, swap R and B)
                let mut rgb_data = Vec::with_capacity(image.data.len() * 3 / 4);
                for chunk in image.data.chunks_exact(4) {
                    rgb_data.push(chunk[2]); // R (was B)
                    rgb_data.push(chunk[1]); // G
                    rgb_data.push(chunk[0]); // B (was R)
                }
                Ok(rgb_data)
            }
            ImageFormat::Grayscale => {
                // Convert grayscale to RGB by replicating channels
                let mut rgb_data = Vec::with_capacity(image.data.len() * 3);
                for &pixel in &image.data {
                    rgb_data.push(pixel); // R
                    rgb_data.push(pixel); // G
                    rgb_data.push(pixel); // B
                }
                Ok(rgb_data)
            }
        }
    }

    /// Resize image data to target size
    fn resize_image(&self, data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        if width == self.input_size.0 && height == self.input_size.1 {
            return Ok(data.to_vec());
        }

        // Create ImageBuffer from raw data
        let img =
            ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data.to_vec()).ok_or_else(|| {
                DetectionError::preprocessing("Failed to create image buffer".to_string())
            })?;

        // Resize using image crate
        let resized = image::imageops::resize(
            &img,
            self.input_size.0,
            self.input_size.1,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(resized.into_raw())
    }

    /// Convert RGB image data to tensor format
    fn to_tensor(&self, data: &[u8]) -> Result<Array4<f32>> {
        let (width, height) = self.input_size;
        let channels = 3;

        // Create tensor: [1, 3, height, width]
        let mut tensor = Array4::<f32>::zeros((1, channels, height as usize, width as usize));

        // Fill tensor with pixel data (HWC -> CHW)
        for y in 0..height {
            for x in 0..width {
                let pixel_idx = ((y * width + x) * 3) as usize;

                if pixel_idx + 2 >= data.len() {
                    return Err(DetectionError::preprocessing(
                        "Pixel index out of bounds".to_string(),
                    ));
                }

                // Convert u8 to f32 and normalize to [0, 1]
                let r = data[pixel_idx] as f32 / 255.0;
                let g = data[pixel_idx + 1] as f32 / 255.0;
                let b = data[pixel_idx + 2] as f32 / 255.0;

                // Apply normalization if enabled
                let (r, g, b) = if self.normalize {
                    (
                        (r - self.mean[0]) / self.std[0],
                        (g - self.mean[1]) / self.std[1],
                        (b - self.mean[2]) / self.std[2],
                    )
                } else {
                    (r, g, b)
                };

                // Store in CHW format
                tensor[[0, 0, y as usize, x as usize]] = r;
                tensor[[0, 1, y as usize, x as usize]] = g;
                tensor[[0, 2, y as usize, x as usize]] = b;
            }
        }

        Ok(tensor)
    }

    /// Create preprocessor with custom input size
    pub fn with_input_size(input_size: (u32, u32)) -> Self {
        Self::new(input_size)
    }

    /// Get input size
    pub fn input_size(&self) -> (u32, u32) {
        self.input_size
    }
}

/// Utility functions for image processing
pub mod utils {
    use super::*;

    /// Convert image from bytes with format detection
    pub fn image_from_bytes(data: &[u8]) -> Result<ImageData> {
        let img = image::load_from_memory(data)
            .map_err(|e| DetectionError::preprocessing(format!("Failed to decode image: {}", e)))?;

        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();

        Ok(ImageData::new(
            rgb_img.into_raw(),
            width,
            height,
            ImageFormat::RGB,
        ))
    }

    /// Save detection visualization to file
    pub fn save_visualization<P: AsRef<Path>>(
        image: &ImageData,
        _detections: &[crate::types::Detection],
        output_path: P,
    ) -> Result<()> {
        if image.format != ImageFormat::RGB {
            return Err(DetectionError::preprocessing(
                "Visualization only supports RGB images".to_string(),
            ));
        }

        // Create image buffer
        let img =
            ImageBuffer::<Rgb<u8>, _>::from_raw(image.width, image.height, image.data.clone())
                .ok_or_else(|| {
                    DetectionError::preprocessing("Failed to create image buffer".to_string())
                })?;

        // Draw bounding boxes (simplified - would need a drawing library for full implementation)
        // This is a placeholder - in practice you'd use a library like imageproc

        // Save image
        img.save(output_path)
            .map_err(|e| DetectionError::preprocessing(format!("Failed to save image: {}", e)))?;

        Ok(())
    }

    /// Calculate optimal input size maintaining aspect ratio
    pub fn calculate_input_size(
        original_width: u32,
        original_height: u32,
        target_size: u32,
    ) -> (u32, u32) {
        let aspect_ratio = original_width as f32 / original_height as f32;

        if original_width > original_height {
            let new_width = target_size;
            let new_height = (target_size as f32 / aspect_ratio) as u32;
            (new_width, new_height)
        } else {
            let new_height = target_size;
            let new_width = (target_size as f32 * aspect_ratio) as u32;
            (new_width, new_height)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = ImagePreprocessor::new((640, 640));
        assert_eq!(preprocessor.input_size(), (640, 640));
    }

    #[test]
    fn test_rgb_conversion() {
        let preprocessor = ImagePreprocessor::new((640, 640));

        // Test RGB (no conversion needed)
        let rgb_data = ImageData::new(vec![255, 128, 64, 200, 150, 100], 2, 1, ImageFormat::RGB);

        let converted = preprocessor.convert_to_rgb(&rgb_data).unwrap();
        assert_eq!(converted, vec![255, 128, 64, 200, 150, 100]);

        // Test BGR conversion
        let bgr_data = ImageData::new(vec![64, 128, 255, 100, 150, 200], 2, 1, ImageFormat::BGR);

        let converted = preprocessor.convert_to_rgb(&bgr_data).unwrap();
        assert_eq!(converted, vec![255, 128, 64, 200, 150, 100]);
    }
}
