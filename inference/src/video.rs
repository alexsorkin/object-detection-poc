//! Video processing utilities for real-time detection
//! This module provides video stream processing capabilities using OpenCV

#[cfg(feature = "opencv")]
use opencv::{
    core::{Mat, MatTrait, MatTraitConst, Size, CV_8UC3},
    imgcodecs::{imread, IMREAD_COLOR},
    imgproc::{resize, INTER_LINEAR, cvt_color, COLOR_BGR2RGB},
    videoio::{VideoCapture, VideoCaptureAPI, CAP_ANY},
    prelude::*,
};

use crate::error::{DetectionError, Result};
use crate::types::{ImageData, ImageFormat, DetectionResult, DetectorConfig};
use crate::MilitaryTargetDetector;

use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "opencv")]
/// Real-time video processor for military target detection
pub struct VideoProcessor {
    detector: Arc<Mutex<MilitaryTargetDetector>>,
    input_size: (u32, u32),
    target_fps: f64,
}

#[cfg(feature = "opencv")]
impl VideoProcessor {
    /// Create new video processor
    pub fn new(detector: MilitaryTargetDetector, target_fps: f64, input_size: (u32, u32)) -> Self {
        Self {
            detector: Arc::new(Mutex::new(detector)),
            input_size,
            target_fps,
        }
    }

    /// Process video from camera
    pub fn process_camera(&self, camera_id: i32) -> Result<VideoStream> {
        let mut cap = VideoCapture::new(camera_id, CAP_ANY)
            .map_err(|e| DetectionError::other(format!("Failed to open camera {}: {}", camera_id, e)))?;

        if !cap.is_opened()
            .map_err(|e| DetectionError::other(format!("Camera check failed: {}", e)))? 
        {
            return Err(DetectionError::other("Camera is not opened".to_string()));
        }

        // Set camera properties for optimal performance
        let _ = cap.set(opencv::videoio::CAP_PROP_FRAME_WIDTH, self.input_size.0 as f64);
        let _ = cap.set(opencv::videoio::CAP_PROP_FRAME_HEIGHT, self.input_size.1 as f64);
        let _ = cap.set(opencv::videoio::CAP_PROP_FPS, self.target_fps);

        log::info!("Camera opened successfully (ID: {})", camera_id);

        Ok(VideoStream::new(cap, self.detector.clone(), self.target_fps))
    }

    /// Process video from file
    pub fn process_file(&self, file_path: &str) -> Result<VideoStream> {
        let cap = VideoCapture::from_file(file_path, CAP_ANY)
            .map_err(|e| DetectionError::other(format!("Failed to open video file: {}", e)))?;

        if !cap.is_opened()
            .map_err(|e| DetectionError::other(format!("Video file check failed: {}", e)))? 
        {
            return Err(DetectionError::other("Video file is not opened".to_string()));
        }

        log::info!("Video file opened: {}", file_path);

        Ok(VideoStream::new(cap, self.detector.clone(), self.target_fps))
    }

    /// Create batch processor for multiple video streams
    pub fn create_batch_processor(&self, max_concurrent: usize) -> BatchVideoProcessor {
        BatchVideoProcessor::new(self.detector.clone(), max_concurrent, self.target_fps)
    }
}

    /// Process video from file
    pub fn process_file(&self, file_path: &str) -> Result<VideoStream> {
        let cap = VideoCapture::from_file(file_path, CAP_ANY)
            .map_err(|e| DetectionError::other(format!("Failed to open video file: {}", e)))?;

        if !cap.is_opened()
            .map_err(|e| DetectionError::other(format!("Video file check failed: {}", e)))? 
        {
            return Err(DetectionError::other("Video file is not opened".to_string()));
        }

        log::info!("Video file opened: {}", file_path);

        Ok(VideoStream::new(cap, self.detector.clone(), self.target_fps))
    }

    /// Create batch processor for multiple video streams
    pub fn create_batch_processor(&self, max_concurrent: usize) -> BatchVideoProcessor {
        BatchVideoProcessor::new(self.detector.clone(), max_concurrent, self.target_fps)
    }
}

#[cfg(feature = "opencv")]
/// Video stream processor
pub struct VideoStream {
    cap: VideoCapture,
    detector: Arc<Mutex<MilitaryTargetDetector>>,
    target_fps: f64,
    frame_interval: Duration,
}

#[cfg(feature = "opencv")]
impl VideoStream {
    fn new(cap: VideoCapture, detector: Arc<Mutex<MilitaryTargetDetector>>, target_fps: f64) -> Self {
        let frame_interval = Duration::from_secs_f64(1.0 / target_fps);
        
        Self {
            cap,
            detector,
            target_fps,
            frame_interval,
        }
    }

    /// Process video stream with callback for each detection result
    pub fn process_with_callback<F>(&mut self, mut callback: F) -> Result<VideoStats>
    where
        F: FnMut(DetectionResult, &Mat) -> bool, // Return false to stop processing
    {
        let mut stats = VideoStats::new();
        let mut frame = Mat::default();
        
        loop {
            let frame_start = Instant::now();

            // Read frame
            if !self.cap.read(&mut frame)
                .map_err(|e| DetectionError::other(format!("Frame read failed: {}", e)))? 
            {
                break; // End of video
            }

            if frame.empty()
                .map_err(|e| DetectionError::other(format!("Frame empty check failed: {}", e)))? 
            {
                break;
            }

            stats.total_frames += 1;

            // Convert frame to detection format
            let image_data = self.mat_to_image_data(&frame)?;
            
            // Run detection (need to lock mutex)
            let detection_start = Instant::now();
            let result = {
                let mut detector = self.detector.lock().unwrap();
                detector.detect(&image_data)?
            };
            let detection_time = detection_start.elapsed();

            stats.total_detections += result.count();
            stats.total_detection_time += detection_time;

            // Call user callback
            if !callback(result, &frame) {
                break; // User requested stop
            }

            // Maintain target FPS
            let frame_time = frame_start.elapsed();
            if frame_time < self.frame_interval {
                thread::sleep(self.frame_interval - frame_time);
            }

            stats.total_processing_time += frame_start.elapsed();
        }

        stats.calculate_averages();
        Ok(stats)
    }

    /// Process video and collect all results
    pub fn process_all(&mut self) -> Result<(Vec<DetectionResult>, VideoStats)> {
        let mut results = Vec::new();
        
        let stats = self.process_with_callback(|result, _| {
            results.push(result);
            true // Continue processing
        })?;

        Ok((results, stats))
    }

    /// Convert OpenCV Mat to ImageData
    fn mat_to_image_data(&self, mat: &Mat) -> Result<ImageData> {
        // Convert BGR to RGB
        let mut rgb_mat = Mat::default();
        cvt_color(mat, &mut rgb_mat, COLOR_BGR2RGB, 0)
            .map_err(|e| DetectionError::preprocessing(format!("Color conversion failed: {}", e)))?;

        // Get image data
        let size = rgb_mat.size()
            .map_err(|e| DetectionError::preprocessing(format!("Mat size failed: {}", e)))?;
        
        let width = size.width as u32;
        let height = size.height as u32;

        // Convert to byte vector
        let data = rgb_mat.data_bytes()
            .map_err(|e| DetectionError::preprocessing(format!("Mat data extraction failed: {}", e)))?
            .to_vec();

        Ok(ImageData::new(data, width, height, ImageFormat::RGB))
    }
}

#[cfg(feature = "opencv")]
/// Batch video processor for multiple concurrent streams
pub struct BatchVideoProcessor {
    detector: Arc<Mutex<MilitaryTargetDetector>>,
    max_concurrent: usize,
    target_fps: f64,
}

#[cfg(feature = "opencv")]
impl BatchVideoProcessor {
    fn new(detector: Arc<Mutex<MilitaryTargetDetector>>, max_concurrent: usize, target_fps: f64) -> Self {
        Self {
            detector,
            max_concurrent,
            target_fps,
        }
    }

    /// Process multiple video sources concurrently
    pub fn process_sources(&self, sources: Vec<VideoSource>) -> Result<Vec<BatchResult>> {
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        // Limit concurrent processing
        for (i, source) in sources.into_iter().enumerate().take(self.max_concurrent) {
            let detector = self.detector.clone();
            let sender = tx.clone();
            let target_fps = self.target_fps;

            let handle = thread::spawn(move || {
                let result = process_single_source(source, detector, target_fps);
                let _ = sender.send((i, result));
            });

            handles.push(handle);
        }

        drop(tx); // Close sender

        // Collect results
        let mut results = Vec::new();
        while let Ok((index, result)) = rx.recv() {
            results.push(BatchResult { index, result });
        }

        // Wait for all threads to complete
        for handle in handles {
            let _ = handle.join();
        }

        // Sort results by index
        results.sort_by_key(|r| r.index);

        Ok(results)
    }
}

#[cfg(feature = "opencv")]
fn process_single_source(
    source: VideoSource,
    detector: Arc<Mutex<MilitaryTargetDetector>>,
    target_fps: f64,
) -> Result<(VideoStats, Vec<DetectionResult>)> {
    // Create a temporary VideoStream directly
    let cap = match &source {
        VideoSource::Camera(id) => VideoCapture::new(*id, CAP_ANY)
            .map_err(|e| DetectionError::other(format!("Failed to open camera: {}", e)))?,
        VideoSource::File(path) => VideoCapture::from_file(path, CAP_ANY)
            .map_err(|e| DetectionError::other(format!("Failed to open video file: {}", e)))?,
    };
    
    let mut stream = VideoStream::new(cap, detector, target_fps);
    stream.process_all()
}

/// Video source enumeration
#[derive(Debug, Clone)]
pub enum VideoSource {
    Camera(i32),
    File(String),
}

/// Batch processing result
#[derive(Debug)]
pub struct BatchResult {
    pub index: usize,
    pub result: Result<(VideoStats, Vec<DetectionResult>)>,
}

/// Video processing statistics
#[derive(Debug, Clone)]
pub struct VideoStats {
    pub total_frames: u64,
    pub total_detections: usize,
    pub total_processing_time: Duration,
    pub total_detection_time: Duration,
    pub average_fps: f64,
    pub average_detection_time_ms: f64,
    pub detections_per_second: f64,
}

impl VideoStats {
    fn new() -> Self {
        Self {
            total_frames: 0,
            total_detections: 0,
            total_processing_time: Duration::ZERO,
            total_detection_time: Duration::ZERO,
            average_fps: 0.0,
            average_detection_time_ms: 0.0,
            detections_per_second: 0.0,
        }
    }

    fn calculate_averages(&mut self) {
        if self.total_frames > 0 {
            self.average_fps = self.total_frames as f64 / self.total_processing_time.as_secs_f64();
            self.average_detection_time_ms = self.total_detection_time.as_secs_f64() * 1000.0 / self.total_frames as f64;
            self.detections_per_second = self.total_detections as f64 / self.total_processing_time.as_secs_f64();
        }
    }
}

// Utility functions for non-OpenCV builds
#[cfg(not(feature = "opencv"))]
pub fn create_dummy_processor() -> Result<()> {
    Err(DetectionError::config(
        "Video processing requires OpenCV feature to be enabled".to_string()
    ))
}