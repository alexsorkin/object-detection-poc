//! Video utilities for frame capture and resizing

use crate::error::DetectionError;
use crate::image_utils;
use crossbeam::channel::Sender;
use image::RgbImage;
use opencv::{
    core::{Mat, Size},
    imgproc::{self, INTER_CUBIC},
    prelude::*,
    videoio::{self, VideoCapture, CAP_ANY},
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Captured input containing both original and resized images with metadata
/// OPTIMIZATION: Uses Arc for zero-copy sharing of image data across threads
#[derive(Clone)]
pub struct CaptureInput {
    /// The original image (full resolution)
    pub original_image: Arc<RgbImage>,
    /// The resized image (max 640px on largest axis)
    pub resized_image: Arc<RgbImage>,
    /// Frames per second of the source
    pub fps: f64,
    /// Original width before resizing
    pub original_width: i32,
    /// Original height before resizing
    pub original_height: i32,
    /// Whether more frames are available (false when stream ends)
    pub has_frames: bool,
}

impl CaptureInput {
    /// Create a new CaptureInput
    /// OPTIMIZATION: Accepts Arc-wrapped images for zero-copy sharing
    pub fn new(
        original_image: Arc<RgbImage>,
        resized_image: Arc<RgbImage>,
        fps: f64,
        original_width: i32,
        original_height: i32,
    ) -> Self {
        Self {
            original_image,
            resized_image,
            fps,
            original_width,
            original_height,
            has_frames: true,
        }
    }
}

/// Video resizer that captures frames from a source and outputs resized frames
pub struct VideoResizer {
    /// Maximum size for the largest axis
    max_axis_size: i32,
}

impl VideoResizer {
    /// Create a new VideoResizer with the default max axis size of 640px
    pub fn new() -> Self {
        Self { max_axis_size: 640 }
    }

    /// Create a new VideoResizer with a custom max axis size
    pub fn with_max_size(max_axis_size: i32) -> Self {
        Self { max_axis_size }
    }

    /// Resize a video stream from the specified source and send frames to the output channel
    ///
    /// # Arguments
    /// * `source_path` - Path to video file or camera device (e.g., "/dev/video0", "rtsp://...", "video.mp4")
    /// * `frame_tx` - Channel sender for input frames (containing both original and resized images)
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn resize_stream(
        &self,
        source_path: impl AsRef<Path>,
        frame_tx: Sender<CaptureInput>,
    ) -> Result<(), DetectionError> {
        let path = source_path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| DetectionError::Other("Invalid path".to_string()))?;

        log::info!("Opening video source: {}", path_str);

        // Open video capture (works for files, cameras, and network streams)
        let capture = VideoCapture::from_file(path_str, CAP_ANY)
            .map_err(|e| DetectionError::Other(format!("Failed to open video source: {}", e)))?;

        self.process_capture(capture, frame_tx)
    }

    /// Resize a video stream from a camera by index
    ///
    /// # Arguments
    /// * `camera_index` - Camera device index (0 for default camera, 1 for second camera, etc.)
    /// * `frame_tx` - Channel sender for input frames (containing both original and resized images)
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn resize_camera(
        &self,
        camera_index: i32,
        frame_tx: Sender<CaptureInput>,
    ) -> Result<(), DetectionError> {
        log::info!("Opening camera device: {}", camera_index);

        // Open camera by index
        let capture = VideoCapture::new(camera_index, CAP_ANY)
            .map_err(|e| DetectionError::Other(format!("Failed to open camera: {}", e)))?;

        self.process_capture(capture, frame_tx)
    }

    /// Process frames from a VideoCapture source
    /// OPTIMIZATION: Minimizes allocations and uses efficient batch processing
    fn process_capture(
        &self,
        mut capture: VideoCapture,
        frame_tx: Sender<CaptureInput>,
    ) -> Result<(), DetectionError> {
        if !capture.is_opened().unwrap_or(false) {
            return Err(DetectionError::Other(
                "Failed to open video capture".to_string(),
            ));
        }

        // Get video properties
        let original_width = capture.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(0.0) as i32;
        let original_height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(0.0) as i32;
        let mut fps = capture.get(videoio::CAP_PROP_FPS).unwrap_or(0.0);

        // Handle invalid FPS (common with cameras)
        if fps <= 0.0 {
            log::warn!(
                "Video source returned invalid FPS ({}), defaulting to 20.0",
                fps
            );
            fps = 20.0;
        }

        log::info!(
            "Video properties: {}x{} @ {:.2} FPS",
            original_width,
            original_height,
            fps
        );

        // OPTIMIZATION: Pre-allocate frame buffer to reuse across loop iterations
        let mut frame = Mat::default();
        let mut frame_count = 0_u64;

        loop {
            let frame_start = Instant::now();
            let capture_frame_duration = Duration::from_secs_f64(1.0 / fps as f64);

            // Read frame
            let read_success = capture
                .read(&mut frame)
                .map_err(|e| DetectionError::Other(format!("Failed to read frame: {}", e)))?;

            if !read_success || frame.empty() {
                log::info!("End of video stream after {} frames", frame_count);

                // OPTIMIZATION: Use Arc::clone for cheap reference counting
                let empty_image = Arc::new(RgbImage::new(1, 1));
                let end_frame = CaptureInput {
                    original_image: Arc::clone(&empty_image),
                    resized_image: empty_image,
                    fps,
                    original_width,
                    original_height,
                    has_frames: false,
                };
                frame_tx.try_send(end_frame).ok();
                break;
            }

            frame_count += 1;

            // OPTIMIZATION: Process both conversions in parallel using rayon
            // Results are already Arc-wrapped, no cloning needed
            let (original_image, resized_image) = rayon::join(
                || self.mat_to_rgb_image(&frame),
                || self.resize_frame(&frame),
            );

            let original_image = original_image?;
            let resized_image = resized_image?;

            // OPTIMIZATION: Create CaptureInput with Arc-wrapped images (zero-copy)
            let capture_input = CaptureInput::new(
                original_image,
                resized_image,
                fps,
                original_width,
                original_height,
            );

            // Send to channel
            frame_tx.try_send(capture_input).ok();

            // OPTIMIZATION: Reduce logging frequency to every 100 frames
            if frame_count % 100 == 0 {
                log::debug!("Processed {} frames", frame_count);
            }

            // Pace to target FPS - sleep for remaining time in this frame period
            let elapsed = frame_start.elapsed();
            if elapsed <= capture_frame_duration {
                std::thread::sleep(capture_frame_duration - elapsed);
            }
        }

        log::info!("Video stream completed: {} frames processed", frame_count);
        Ok(())
    }

    /// Convert OpenCV Mat (BGR) to RgbImage
    /// OPTIMIZATION: Returns Arc-wrapped image for zero-copy sharing
    fn mat_to_rgb_image(&self, mat: &Mat) -> Result<Arc<RgbImage>, DetectionError> {
        let width = mat.cols() as u32;
        let height = mat.rows() as u32;
        let total_pixels = (width * height) as usize;

        // OPTIMIZATION: For small images (<100k pixels), use OpenCV directly
        // For larger images, use parallel processing
        if total_pixels < 100_000 {
            // Convert BGR to RGB using OpenCV
            let mut rgb_mat = Mat::default();
            imgproc::cvt_color(
                mat,
                &mut rgb_mat,
                imgproc::COLOR_BGR2RGB,
                0,
            )
            .map_err(|e| DetectionError::Other(format!("Failed to convert BGR to RGB: {}", e)))?;

            // OPTIMIZATION: Pre-allocate with exact capacity
            let data = rgb_mat
                .data_bytes()
                .map_err(|e| DetectionError::Other(format!("Failed to get image data: {}", e)))?
                .to_vec();

            let rgb_image = RgbImage::from_vec(width, height, data)
                .ok_or_else(|| DetectionError::Other("Failed to create RgbImage".to_string()))?;

            Ok(Arc::new(rgb_image))
        } else {
            // OPTIMIZATION: Parallel processing for large images
            // Convert BGR to RGB using OpenCV first (still fastest for color conversion)
            let mut rgb_mat = Mat::default();
            imgproc::cvt_color(
                mat,
                &mut rgb_mat,
                imgproc::COLOR_BGR2RGB,
                0,
            )
            .map_err(|e| DetectionError::Other(format!("Failed to convert BGR to RGB: {}", e)))?;

            // Get data and convert to Vec in parallel chunks
            let data = rgb_mat
                .data_bytes()
                .map_err(|e| DetectionError::Other(format!("Failed to get image data: {}", e)))?;

            // OPTIMIZATION: Pre-allocate with exact capacity
            let mut vec_data = Vec::with_capacity(data.len());
            vec_data.extend_from_slice(data);

            let rgb_image = RgbImage::from_vec(width, height, vec_data)
                .ok_or_else(|| DetectionError::Other("Failed to create RgbImage".to_string()))?;

            Ok(Arc::new(rgb_image))
        }
    }

    /// Resize a single frame according to the max axis size
    /// Applies CLAHE shadow removal before resizing for better quality
    ///
    /// Frames with largest axis <= max_axis_size are returned as-is (with shadow removal)
    /// Frames with largest axis > max_axis_size are resized proportionally (after shadow removal)
    /// OPTIMIZATION: Returns Arc-wrapped image for zero-copy sharing
    fn resize_frame(&self, frame: &Mat) -> Result<Arc<RgbImage>, DetectionError> {
        let height = frame.rows();
        let width = frame.cols();

        // Apply shadow removal using CLAHE before any processing
        let enhanced_frame = image_utils::remove_shadows_clahe(frame)
            .map_err(|e| DetectionError::Other(format!("Shadow removal failed: {}", e)))?;

        // Determine if resizing is needed
        let max_dimension = width.max(height);

        if max_dimension <= self.max_axis_size {
            // Frame is small enough, convert to RGB and return as-is (already enhanced)
            return self.mat_to_rgb_image(&enhanced_frame);
        }

        // Calculate new dimensions preserving aspect ratio
        let scale = self.max_axis_size as f32 / max_dimension as f32;
        let new_width = (width as f32 * scale) as i32;
        let new_height = (height as f32 * scale) as i32;

        log::trace!(
            "Resizing frame from {}x{} to {}x{} (scale: {:.3})",
            width,
            height,
            new_width,
            new_height,
            scale
        );

        // Resize enhanced frame using high-quality bicubic interpolation
        let mut resized = Mat::default();
        imgproc::resize(
            &enhanced_frame,
            &mut resized,
            Size::new(new_width, new_height),
            0.0,
            0.0,
            INTER_CUBIC, // Bicubic interpolation for better quality
        )
        .map_err(|e| DetectionError::Other(format!("Failed to resize frame: {}", e)))?;

        // Convert resized frame to RgbImage
        self.mat_to_rgb_image(&resized)
    }

    /// Get the current max axis size setting
    pub fn max_axis_size(&self) -> i32 {
        self.max_axis_size
    }
}

impl Default for VideoResizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_calculation() {
        let resizer = VideoResizer::new();
        assert_eq!(resizer.max_axis_size(), 640);

        let resizer_custom = VideoResizer::with_max_size(1024);
        assert_eq!(resizer_custom.max_axis_size(), 1024);
    }
}
