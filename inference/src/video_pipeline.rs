use crate::frame_pipeline::{DetectionPipeline, TileDetection};
/// Video processing pipeline with async detection and temporal tracking
///
/// Handles real-time video streams with:
/// - Async frame detection with backpressure
/// - Multiple tracking algorithms (Kalman filter or ByteTrack)
/// - Extrapolation when detection latency is high
use crate::tracking::{TrackingConfig, UnifiedTracker};
use crossbeam::channel::{bounded, Receiver, Sender};

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Frame with metadata
#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub sequence: u64,
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

/// Internal worker commands
#[derive(Debug)]
enum WorkerCommand {
    ProcessFrame(Frame),
    AdvanceTracks,
}

/// Video pipeline configuration
pub struct VideoPipelineConfig {
    /// Maximum latency before switching to extrapolation-only mode (ms)
    pub max_latency_ms: u64,
    /// Tracking configuration (Kalman or ByteTrack)
    pub tracking_config: TrackingConfig,
    /// Frame buffer size (number of frames to queue)
    pub buffer_size: usize,
}

impl Default for VideoPipelineConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 500,
            tracking_config: TrackingConfig::default(),
            buffer_size: 2,
        }
    }
}

/// Video processing pipeline with async detection and temporal tracking
pub struct VideoPipeline {
    command_tx: Sender<WorkerCommand>,
    result_rx: Receiver<FrameResult>,
    _worker_handle: thread::JoinHandle<()>,
}

impl VideoPipeline {
    /// Create new video processing pipeline
    pub fn new(detection_pipeline: Arc<DetectionPipeline>, config: VideoPipelineConfig) -> Self {
        let (command_tx, command_rx) = bounded::<WorkerCommand>(config.buffer_size);
        let (result_tx, result_rx) = bounded::<FrameResult>(config.buffer_size);

        let tracker = UnifiedTracker::new(config.tracking_config);
        let detector = Arc::clone(&detection_pipeline);
        let max_latency = Duration::from_millis(config.max_latency_ms);

        let worker_handle = thread::spawn(move || {
            Self::worker_loop(command_rx, result_tx, tracker, detector, max_latency);
        });

        Self {
            command_tx,
            result_rx,
            _worker_handle: worker_handle,
        }
    }

    /// Worker thread: processes frames and manages unified tracker
    fn worker_loop(
        command_rx: Receiver<WorkerCommand>,
        result_tx: Sender<FrameResult>,
        mut tracker: UnifiedTracker,
        detector: Arc<DetectionPipeline>,
        _max_latency: Duration,
    ) {
        let mut last_process_time = Instant::now();
        let mut frames_processed = 0_u64;
        let frames_extrapolated = 0_u64;

        log::info!(
            "Video pipeline worker started with {} tracker",
            tracker.method()
        );

        while let Ok(command) = command_rx.recv() {
            match command {
                WorkerCommand::AdvanceTracks => {
                    // Just advance tracker predictions without processing
                    tracker.update(&[], 0.04); // Assume ~25ms between frames for dt
                    log::debug!("Advanced tracker predictions (dropped frame)");
                    continue;
                }
                WorkerCommand::ProcessFrame(frame) => {
                    let now = Instant::now();
                    let dt = now.duration_since(last_process_time).as_secs_f32();

                    // Convert frame data to image for processing
                    let img_data = frame.data.clone();
                    let width = frame.width;
                    let height = frame.height;

                    // Create an RgbImage from raw data (assuming RGB format)
                    let image = match image::RgbImage::from_raw(width, height, img_data) {
                        Some(img) => img,
                        None => {
                            log::error!("Failed to create image from frame data");
                            continue;
                        }
                    };

                    // PROCESSING MODE: Run detection pipeline with tracker callback
                    let process_start = Instant::now();

                    // Use shared reference for callback results
                    let mut tracked_detections = Vec::new();

                    {
                        // Create callback that updates tracker when detection completes
                        let tracker_ref = &mut tracker;
                        let tracked_detections_ref = &mut tracked_detections;
                        let callback = |detections: &[TileDetection], _processing_time: f32| {
                            // Update tracker immediately when detection completes
                            *tracked_detections_ref = tracker_ref.update(detections, dt);
                        };
                        match detector.process_with_callback_blocking(&image, callback) {
                            Ok(_) => {
                                let processing_time =
                                    process_start.elapsed().as_secs_f32() * 1000.0;

                                let result = FrameResult {
                                    frame_id: frame.sequence,
                                    timestamp: now,
                                    detections: tracked_detections.clone(),
                                    is_extrapolated: false,
                                    processing_time_ms: processing_time,
                                    latency_ms: 0.0, // No latency in command-based processing
                                };

                                frames_processed += 1;
                                if let Err(e) = result_tx.send(result) {
                                    log::warn!("Failed to send detection result: {}", e);
                                    break;
                                }

                                log::debug!(
                                    "Frame {} PROCESSED (processing: {:.1}ms, {} tracks)",
                                    frame.sequence,
                                    processing_time,
                                    tracker.num_tracks()
                                );
                            }
                            Err(e) => {
                                log::error!("Detection pipeline callback failed: {:?}", e);
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
            }
        }

        log::info!("Real-time pipeline worker stopped");
    }

    /// Submit frame for processing
    pub fn submit_frame(&self, frame: Frame) -> Result<(), String> {
        self.command_tx
            .send(WorkerCommand::ProcessFrame(frame))
            .map_err(|e| format!("Failed to submit frame: {}", e))
    }

    /// Try to submit frame (non-blocking, returns false if buffer full)
    pub fn try_submit_frame(&self, frame: Frame) -> bool {
        self.command_tx
            .try_send(WorkerCommand::ProcessFrame(frame))
            .is_ok()
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

    /// Advance tracker predictions for dropped frames (async, non-blocking)
    /// This advances the tracker's motion models without processing a frame
    pub fn advance_tracks(&self) {
        // Send a command to advance tracker predictions without frame processing
        let _ = self.command_tx.try_send(WorkerCommand::AdvanceTracks);
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
                data: vec![0u8; 640 * 480 * 3], // RGB dummy data
                width: 640,
                height: 480,
                sequence: i,
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
