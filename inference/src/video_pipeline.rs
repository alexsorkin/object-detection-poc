use crate::frame_pipeline::{DetectionPipeline, TileDetection};
/// Video processing pipeline with async detection and Kalman filter extrapolation
///
/// Handles real-time video streams with:
/// - Async frame detection with backpressure
/// - Kalman filter for temporal tracking
/// - Extrapolation when detection latency is high
use crate::kalman_tracker::{KalmanConfig, MultiObjectTracker};
use crossbeam::channel::{bounded, Receiver, Sender};
use image::RgbImage;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Frame with metadata
#[derive(Clone)]
pub struct Frame {
    pub frame_id: u64,
    pub image: RgbImage,
    pub timestamp: Instant,
}

/// Detection result with frame info
#[derive(Clone)]
pub struct FrameResult {
    pub frame_id: u64,
    pub timestamp: Instant,
    pub detections: Vec<TileDetection>,
    pub is_extrapolated: bool,
    pub processing_time_ms: f32,
    pub latency_ms: f32,
}

/// Video pipeline configuration
pub struct VideoPipelineConfig {
    /// Maximum latency before switching to extrapolation-only mode (ms)
    pub max_latency_ms: u64,
    /// Kalman filter configuration
    pub kalman_config: KalmanConfig,
    /// Frame buffer size (number of frames to queue)
    pub buffer_size: usize,
}

impl Default for VideoPipelineConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 500,
            kalman_config: KalmanConfig::default(),
            buffer_size: 10,
        }
    }
}

/// Video processing pipeline with async detection and temporal tracking
pub struct VideoPipeline {
    frame_tx: Sender<Frame>,
    result_rx: Receiver<FrameResult>,
    _worker_handle: thread::JoinHandle<()>,
}

impl VideoPipeline {
    /// Create new video processing pipeline
    pub fn new(detection_pipeline: Arc<DetectionPipeline>, config: VideoPipelineConfig) -> Self {
        let (frame_tx, frame_rx) = bounded::<Frame>(config.buffer_size);
        let (result_tx, result_rx) = bounded::<FrameResult>(config.buffer_size * 2);

        let worker_handle = thread::spawn(move || {
            Self::worker_thread(detection_pipeline, frame_rx, result_tx, config);
        });

        Self {
            frame_tx,
            result_rx,
            _worker_handle: worker_handle,
        }
    }

    /// Worker thread: processes frames and manages Kalman tracker
    fn worker_thread(
        pipeline: Arc<DetectionPipeline>,
        frame_rx: Receiver<Frame>,
        result_tx: Sender<FrameResult>,
        config: VideoPipelineConfig,
    ) {
        let mut tracker = MultiObjectTracker::new(config.kalman_config.clone());
        let mut last_process_time = Instant::now();
        let mut frames_processed = 0_u64;
        let mut frames_extrapolated = 0_u64;

        log::info!("Video pipeline worker started");

        while let Ok(mut frame) = frame_rx.recv() {
            let now = Instant::now();
            let mut latency = now.duration_since(frame.timestamp).as_millis() as f32;

            // CRITICAL: If we're badly backed up, drain old frames and use the newest one
            if latency > config.max_latency_ms as f32 * 2.0 {
                let mut skipped = 0;
                // Drain the queue and find the newest frame
                while let Ok(newer_frame) = frame_rx.try_recv() {
                    frame = newer_frame;
                    skipped += 1;
                    latency = now.duration_since(frame.timestamp).as_millis() as f32;
                    // If we found a recent enough frame, stop draining
                    if latency < config.max_latency_ms as f32 {
                        break;
                    }
                }
                if skipped > 0 {
                    log::warn!("Drained {} stale frames from queue", skipped);
                }
            }

            let dt = now.duration_since(last_process_time).as_secs_f32();

            // Check if we're falling too far behind
            let should_extrapolate = latency > config.max_latency_ms as f32;

            if should_extrapolate {
                // EXTRAPOLATION MODE: Use Kalman predictions only
                tracker.update(&[], dt); // Predict without measurements
                let predictions = tracker.get_predictions();

                let result = FrameResult {
                    frame_id: frame.frame_id,
                    timestamp: frame.timestamp,
                    detections: predictions,
                    is_extrapolated: true,
                    processing_time_ms: 0.0,
                    latency_ms: latency,
                };

                frames_extrapolated += 1;
                if let Err(e) = result_tx.send(result) {
                    log::warn!("Failed to send extrapolated result: {}", e);
                    break;
                }

                log::debug!(
                    "Frame {} EXTRAPOLATED (latency: {:.1}ms, {} tracks)",
                    frame.frame_id,
                    latency,
                    tracker.num_tracks()
                );
            } else {
                // PROCESSING MODE: Run detection pipeline
                let process_start = Instant::now();

                match pipeline.process(&frame.image) {
                    Ok(output) => {
                        let processing_time = process_start.elapsed().as_secs_f32() * 1000.0;

                        // Update tracker with real detections
                        tracker.update(&output.detections, dt);

                        let result = FrameResult {
                            frame_id: frame.frame_id,
                            timestamp: frame.timestamp,
                            detections: output.detections.clone(),
                            is_extrapolated: false,
                            processing_time_ms: processing_time,
                            latency_ms: latency,
                        };

                        frames_processed += 1;
                        if let Err(e) = result_tx.send(result) {
                            log::warn!("Failed to send detection result: {}", e);
                            break;
                        }

                        log::debug!(
                            "Frame {} PROCESSED (latency: {:.1}ms, processing: {:.1}ms, {} detections, {} tracks)",
                            frame.frame_id,
                            latency,
                            processing_time,
                            output.detections.len(),
                            tracker.num_tracks()
                        );
                    }
                    Err(e) => {
                        log::error!("Detection pipeline error: {}", e);
                    }
                }
            }

            last_process_time = now;

            // Log statistics every 100 frames
            if (frames_processed + frames_extrapolated) % 100 == 0 {
                let total = frames_processed + frames_extrapolated;
                let extrapolation_rate = frames_extrapolated as f32 / total as f32 * 100.0;
                log::info!(
                    "Pipeline stats: {} processed, {} extrapolated ({:.1}%)",
                    frames_processed,
                    frames_extrapolated,
                    extrapolation_rate
                );
            }
        }

        log::info!("Real-time pipeline worker stopped");
    }

    /// Submit frame for processing
    pub fn submit_frame(&self, frame: Frame) -> Result<(), String> {
        self.frame_tx
            .send(frame)
            .map_err(|e| format!("Failed to submit frame: {}", e))
    }

    /// Try to submit frame (non-blocking, returns false if buffer full)
    pub fn try_submit_frame(&self, frame: Frame) -> bool {
        self.frame_tx.try_send(frame).is_ok()
    }

    /// Get next result (blocking)
    pub fn get_result(&self) -> Result<FrameResult, String> {
        self.result_rx
            .recv()
            .map_err(|e| format!("Failed to receive result: {}", e))
    }

    /// Try to get next result (non-blocking)
    pub fn try_get_result(&self) -> Option<FrameResult> {
        self.result_rx.try_recv().ok()
    }

    /// Get result with timeout
    pub fn get_result_timeout(&self, timeout: Duration) -> Result<FrameResult, String> {
        self.result_rx
            .recv_timeout(timeout)
            .map_err(|e| format!("Timeout waiting for result: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector_trait::DetectorType;
    use crate::frame_executor::{ExecutorConfig, FrameExecutor};
    use crate::frame_pipeline::{DetectionPipeline, PipelineConfig};
    use crate::types::DetectorConfig;
    use std::sync::Arc;

    #[test]
    #[ignore] // Requires model files
    fn test_realtime_pipeline() {
        env_logger::init();

        // Create frame executor
        let detector_config = DetectorConfig::default();
        let executor_config = ExecutorConfig { max_queue_depth: 2 };

        let frame_executor = Arc::new(
            FrameExecutor::new(DetectorType::RTDETR, detector_config, executor_config).unwrap(),
        );

        // Create pipeline
        let pipeline_config = PipelineConfig::default();
        let pipeline = Arc::new(DetectionPipeline::new(frame_executor, pipeline_config));

        // Create video pipeline
        let config = VideoPipelineConfig::default();
        let video_pipeline = VideoPipeline::new(pipeline, config);

        // Submit test frames
        for i in 0..10 {
            let frame = Frame {
                frame_id: i,
                image: RgbImage::new(640, 480),
                timestamp: Instant::now(),
            };
            video_pipeline.submit_frame(frame).unwrap();
        }

        // Get results
        for _ in 0..10 {
            let result = video_pipeline
                .get_result_timeout(Duration::from_secs(5))
                .unwrap();
            println!(
                "Frame {}: {} detections, extrapolated: {}",
                result.frame_id,
                result.detections.len(),
                result.is_extrapolated
            );
        }
    }
}
