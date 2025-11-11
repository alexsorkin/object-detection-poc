use crate::frame_pipeline::{DetectionPipeline, PipelineOutput, TileDetection};
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
    pub duplicates_removed: usize,
    pub nested_removed: usize,
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

        // Channel for sending detection outputs from callback to detection loop
        let (detection_output_tx, detection_output_rx) =
            bounded::<(u64, Instant, PipelineOutput)>(config.buffer_size);

        let tracker = UnifiedTracker::new(config.tracking_config);
        let detector = Arc::clone(&detection_pipeline);
        let max_latency = Duration::from_millis(config.max_latency_ms);

        // Start detection loop thread with tracker - it sends results directly to result_tx
        let result_tx_clone = result_tx.clone();
        let _detection_handle = thread::spawn(move || {
            Self::detection_loop(detection_output_rx, result_tx_clone, tracker);
        });

        let worker_handle = thread::spawn(move || {
            Self::worker_loop(command_rx, detector, detection_output_tx, max_latency);
        });

        Self {
            command_tx,
            result_rx,
            _worker_handle: worker_handle,
        }
    }

    /// Detection loop: processes detection outputs and advance track commands
    fn detection_loop(
        output_rx: Receiver<(u64, Instant, PipelineOutput)>,
        result_tx: Sender<FrameResult>,
        mut tracker: UnifiedTracker,
    ) {
        log::info!("Detection loop started with {} tracker", tracker.method());
        let mut last_process_time = Instant::now();

        while let Ok((frame_id, timestamp, pipeline_output)) = output_rx.recv() {
            let now = Instant::now();
            let dt = now.duration_since(last_process_time).as_secs_f32();

            let tracked_detections = tracker.update(&pipeline_output.detections, dt);

            // Create frame result with extracted diagnostics
            let result = FrameResult {
                frame_id,
                timestamp,
                detections: tracked_detections,
                is_extrapolated: false,
                processing_time_ms: pipeline_output.pipeline_total_time_ms,
                latency_ms: 0.0, // No latency in command-based processing
                duplicates_removed: pipeline_output.duplicates_removed,
                nested_removed: pipeline_output.nested_removed,
            };

            log::info!(
                "Detection pipeline time: {:.1}ms",
                pipeline_output.pipeline_total_time_ms
            );

            if let Err(e) = result_tx.send(result) {
                log::warn!("Failed to send detection result: {}", e);
                break;
            }

            log::debug!(
                "Frame {} PROCESSED (processing: {:.1}ms, {} tracks)",
                frame_id,
                pipeline_output.pipeline_total_time_ms,
                tracker.num_tracks()
            );

            last_process_time = now;
        }

        log::info!("Detection loop stopped");
    }

    /// Worker thread: handles commands and triggers detection
    fn worker_loop(
        command_rx: Receiver<WorkerCommand>,
        detector: Arc<DetectionPipeline>,
        detection_output_tx: Sender<(u64, Instant, PipelineOutput)>,
        _max_latency: Duration,
    ) {
        let mut frames_processed = 0_u64;
        let frames_extrapolated = 0_u64;

        log::info!("Video pipeline worker started");

        while let Ok(command) = command_rx.recv() {
            match command {
                WorkerCommand::AdvanceTracks => {
                    // TODO: Send advance tracks command to detection loop
                    log::debug!("Advanced tracker predictions (dropped frame)");
                    continue;
                }
                WorkerCommand::ProcessFrame(frame) => {
                    let now = Instant::now();
                    let frame_id = frame.sequence;
                    let timestamp = now;

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

                    // Trigger detection with callback that sends to detection loop
                    let output_tx_clone = detection_output_tx.clone();
                    detector.process_with_callback(&image, move |output: &PipelineOutput| {
                        // Send frame context and pipeline output to detection loop
                        if let Err(e) = output_tx_clone.send((frame_id, timestamp, output.clone()))
                        {
                            log::warn!("Failed to send detection output to detection loop: {}", e);
                        }
                    });

                    frames_processed += 1;

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
