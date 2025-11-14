//! Video utilities for frame capture and resizing

use crate::error::DetectionError;
use crossbeam::channel::Sender;
use opencv::{
    core::{Mat, Size},
    imgproc::{self, INTER_LINEAR},
    prelude::*,
    videoio::{self, VideoCapture, CAP_ANY},
};
use std::path::Path;

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
    /// * `resized_tx` - Channel sender for resized frames
    /// * `original_tx` - Channel sender for original (unresized) frames
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn resize_stream(
        &self,
        source_path: impl AsRef<Path>,
        resized_tx: Sender<Mat>,
        original_tx: Sender<Mat>,
    ) -> Result<(), DetectionError> {
        let path = source_path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| DetectionError::Other("Invalid path".to_string()))?;

        log::info!("Opening video source: {}", path_str);

        // Open video capture (works for files, cameras, and network streams)
        let capture = VideoCapture::from_file(path_str, CAP_ANY)
            .map_err(|e| DetectionError::Other(format!("Failed to open video source: {}", e)))?;

        self.process_capture(capture, resized_tx, original_tx)
    }

    /// Resize a video stream from a camera by index
    ///
    /// # Arguments
    /// * `camera_index` - Camera device index (0 for default camera, 1 for second camera, etc.)
    /// * `resized_tx` - Channel sender for resized frames
    /// * `original_tx` - Channel sender for original (unresized) frames
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn resize_camera(
        &self,
        camera_index: i32,
        resized_tx: Sender<Mat>,
        original_tx: Sender<Mat>,
    ) -> Result<(), DetectionError> {
        log::info!("Opening camera device: {}", camera_index);

        // Open camera by index
        let capture = VideoCapture::new(camera_index, CAP_ANY)
            .map_err(|e| DetectionError::Other(format!("Failed to open camera: {}", e)))?;

        self.process_capture(capture, resized_tx, original_tx)
    }

    /// Process frames from a VideoCapture source
    fn process_capture(
        &self,
        mut capture: VideoCapture,
        resized_tx: Sender<Mat>,
        original_tx: Sender<Mat>,
    ) -> Result<(), DetectionError> {
        if !capture.is_opened().unwrap_or(false) {
            return Err(DetectionError::Other(
                "Failed to open video capture".to_string(),
            ));
        }

        // Get video properties
        let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(0.0) as i32;
        let height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(0.0) as i32;
        let fps = capture.get(videoio::CAP_PROP_FPS).unwrap_or(30.0);

        log::info!("Video properties: {}x{} @ {:.2} FPS", width, height, fps);

        let mut frame = Mat::default();
        let mut frame_count = 0_u64;

        loop {
            // Read frame
            let read_success = capture
                .read(&mut frame)
                .map_err(|e| DetectionError::Other(format!("Failed to read frame: {}", e)))?;

            if !read_success || frame.empty() {
                log::info!("End of video stream after {} frames", frame_count);
                break;
            }

            frame_count += 1;

            // Send original frame to original channel
            let original_frame = frame.try_clone().map_err(|e| {
                DetectionError::Other(format!("Failed to clone original frame: {}", e))
            })?;

            if original_tx.send(original_frame).is_err() {
                log::warn!("Original frame channel closed, stopping video capture");
                break;
            }

            // Resize frame if needed
            let resized_frame = self.resize_frame(&frame)?;

            // Send resized frame to resized channel
            if resized_tx.send(resized_frame).is_err() {
                log::warn!("Resized frame channel closed, stopping video capture");
                break;
            }

            if frame_count % 100 == 0 {
                log::debug!("Processed {} frames", frame_count);
            }
        }

        log::info!("Video stream completed: {} frames processed", frame_count);
        Ok(())
    }

    /// Resize a single frame according to the max axis size
    ///
    /// Frames with largest axis <= max_axis_size are returned as-is
    /// Frames with largest axis > max_axis_size are resized proportionally
    fn resize_frame(&self, frame: &Mat) -> Result<Mat, DetectionError> {
        let height = frame.rows();
        let width = frame.cols();

        // Determine if resizing is needed
        let max_dimension = width.max(height);

        if max_dimension <= self.max_axis_size {
            // Frame is small enough, return as-is (clone to avoid ownership issues)
            return frame
                .try_clone()
                .map_err(|e| DetectionError::Other(format!("Failed to clone frame: {}", e)));
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

        // Resize frame
        let mut resized = Mat::default();
        imgproc::resize(
            frame,
            &mut resized,
            Size::new(new_width, new_height),
            0.0,
            0.0,
            INTER_LINEAR,
        )
        .map_err(|e| DetectionError::Other(format!("Failed to resize frame: {}", e)))?;

        Ok(resized)
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
