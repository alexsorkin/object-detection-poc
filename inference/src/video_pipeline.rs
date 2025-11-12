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
    Shutdown,
}

/// Internal dropped frames commands
#[derive(Debug)]
enum DroppedFrameCommand {
    AdvanceTracks,
    Shutdown,
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
///
/// Features:
/// - Graceful shutdown with cleanup of all worker threads
/// - Automatic shutdown when dropped
/// - Non-blocking shutdown attempts
pub struct VideoPipeline {
    command_tx: Sender<WorkerCommand>,
    dropped_tx: Sender<DroppedFrameCommand>,
    result_rx: Receiver<FrameResult>,
    shutdown_tx: Sender<()>,
    _worker_handle: thread::JoinHandle<()>,
    _dropped_handle: thread::JoinHandle<()>,
    _detection_handle: thread::JoinHandle<()>,
}

impl VideoPipeline {
    /// Create new video processing pipeline
    pub fn new(detection_pipeline: Arc<DetectionPipeline>, config: VideoPipelineConfig) -> Self {
        let (command_tx, command_rx) = bounded::<WorkerCommand>(config.buffer_size);

        let (dropped_tx, dropped_rx) = bounded::<DroppedFrameCommand>(1);
        let (result_tx, result_rx) = bounded::<FrameResult>(1);
        let (output_tx, output_rx) = bounded::<(u64, Instant, PipelineOutput)>(1);

        // Shutdown channels
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
        let shutdown_rx_detection = shutdown_rx.clone();
        let shutdown_rx_dropped = shutdown_rx.clone();

        let tracker = UnifiedTracker::new(config.tracking_config);
        let detector = Arc::clone(&detection_pipeline);
        let max_latency = Duration::from_millis(config.max_latency_ms);

        // Start output loop thread with tracker - it sends results directly to result_tx
        let result_tx_clone = result_tx.clone();
        let tracker_det_clone = tracker.clone();
        let detection_handle = thread::spawn(move || {
            Self::output_loop(
                output_rx,
                result_tx_clone,
                tracker_det_clone,
                shutdown_rx_detection,
            );
        });

        // Start dedicated dropped frames loop
        let tracker_drop_clone = tracker.clone();
        let dropped_handle = thread::spawn(move || {
            Self::dropped_frames_loop(dropped_rx, tracker_drop_clone, shutdown_rx_dropped);
        });

        let worker_handle = thread::spawn(move || {
            Self::worker_loop(command_rx, detector, output_tx, max_latency, shutdown_rx);
        });

        Self {
            command_tx,
            dropped_tx,
            result_rx,
            shutdown_tx,
            _worker_handle: worker_handle,
            _dropped_handle: dropped_handle,
            _detection_handle: detection_handle,
        }
    }

    /// Output loop: processes detection outputs and advance track commands
    fn output_loop(
        output_rx: Receiver<(u64, Instant, PipelineOutput)>,
        result_tx: Sender<FrameResult>,
        mut tracker: UnifiedTracker,
        shutdown_rx: Receiver<()>,
    ) {
        log::info!("Output loop started with {} tracker", tracker.method());
        let mut last_process_time = Instant::now();

        loop {
            crossbeam::select! {
                recv(output_rx) -> msg => {
                    match msg {
                        Ok((frame_id, timestamp, pipeline_output)) => {
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

                            if let Err(e) = result_tx.try_send(result) {
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
                        Err(_) => {
                            log::info!("Detection output channel closed");
                            break;
                        }
                    }
                }
                recv(shutdown_rx) -> _ => {
                    log::info!("Output loop received shutdown signal");
                    break;
                }
            }
        }

        log::info!("Output loop stopped");
    }

    /// Dropped frames loop: handles advancing tracks for dropped frames
    fn dropped_frames_loop(
        dropped_rx: Receiver<DroppedFrameCommand>,
        mut tracker: UnifiedTracker,
        shutdown_rx: Receiver<()>,
    ) {
        log::info!("Dropped frames loop started");
        let mut frames_extrapolated = 0_u64;

        loop {
            crossbeam::select! {
                recv(dropped_rx) -> msg => {
                    match msg {
                        Ok(command) => {
                            match command {
                                DroppedFrameCommand::AdvanceTracks => {
                                    let now = Instant::now();
                                    let timestamp = Instant::now();
                                    let dt = now.duration_since(timestamp).as_secs_f32();

                                    // Update tracker with empty detections to advance predictions
                                    let tracked_predictions = tracker.update(&[], dt);

                                    log::debug!(
                                        "Advanced tracks: {} predictions, dt={:.3}s",
                                        tracked_predictions.len(),
                                        dt
                                    );

                                    log::warn!("Advancing trackers for dropped frame (pipeline busy)");

                                    frames_extrapolated += 1;

                                    // Log statistics every 50 dropped frames
                                    if frames_extrapolated % 50 == 0 {
                                        log::debug!("Dropped frames handled: {}", frames_extrapolated);
                                    }
                                }
                                DroppedFrameCommand::Shutdown => {
                                    log::info!("Dropped frames loop received shutdown command");
                                    break;
                                }
                            }
                        }
                        Err(_) => {
                            log::info!("Dropped frame command channel closed");
                            break;
                        }
                    }
                }
                recv(shutdown_rx) -> _ => {
                    log::info!("Dropped frames loop received shutdown signal");
                    break;
                }
            }
        }

        log::info!(
            "Dropped frames loop stopped (total extrapolated: {})",
            frames_extrapolated
        );
    }

    /// Worker thread: handles commands and triggers detection
    fn worker_loop(
        command_rx: Receiver<WorkerCommand>,
        detector: Arc<DetectionPipeline>,
        output_tx: Sender<(u64, Instant, PipelineOutput)>,
        _max_latency: Duration,
        shutdown_rx: Receiver<()>,
    ) {
        let mut frames_processed = 0_u64;

        log::info!("Video pipeline worker started");

        loop {
            crossbeam::select! {
                recv(command_rx) -> msg => {
                    match msg {
                        Ok(command) => {
                            match command {
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

                                    // Trigger detection with callback that sends to output loop
                                    let output_tx_clone = output_tx.clone();
                                    detector.process_with_callback(&image, move |output: &PipelineOutput| {
                                        // Send frame context and pipeline output to output loop
                                        if let Err(e) = output_tx_clone.try_send((frame_id, timestamp, output.clone()))
                                        {
                                            log::warn!("Failed to send detection output: {}", e);
                                        }
                                    });

                                    frames_processed += 1;

                                    // Log statistics every 100 frames
                                    if frames_processed % 100 == 0 {
                                        log::info!(
                                            "Worker pipeline stats: {} frames processed",
                                            frames_processed
                                        );
                                    }
                                }
                                WorkerCommand::Shutdown => {
                                    log::info!("Worker loop received shutdown command");
                                    break;
                                }
                            }
                        }
                        Err(_) => {
                            log::info!("Command channel closed");
                            break;
                        }
                    }
                }
                recv(shutdown_rx) -> _ => {
                    log::info!("Worker loop received shutdown signal");
                    break;
                }
            }
        }

        log::info!("Video processor pipeline stopped");
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
    pub fn advance_tracks(&self) -> bool {
        // Send a command to advance tracker predictions to the dedicated dropped frames loop
        self.dropped_tx
            .try_send(DroppedFrameCommand::AdvanceTracks)
            .is_ok()
    }

    /// Shutdown the video pipeline gracefully
    /// This will stop all worker threads and close all channels
    pub fn shutdown(&self) {
        log::debug!("Initiating video pipeline shutdown");

        // Send shutdown command to worker
        let _ = self.command_tx.send(WorkerCommand::Shutdown);

        // Send shutdown command to dropped frame loop
        let _ = self.dropped_tx.send(DroppedFrameCommand::Shutdown);

        // Send shutdown signal to all threads
        let _ = self.shutdown_tx.send(());
    }

    /// Try to shutdown the pipeline (non-blocking)
    pub fn try_shutdown(&self) -> bool {
        log::debug!("Attempting non-blocking video pipeline shutdown");

        // Try to send shutdown command to worker
        if self.command_tx.try_send(WorkerCommand::Shutdown).is_err() {
            return false;
        }

        // Try to send shutdown command to dropped frame loop
        if self
            .dropped_tx
            .try_send(DroppedFrameCommand::Shutdown)
            .is_err()
        {
            return false;
        }

        // Try to send shutdown signal
        self.shutdown_tx.try_send(()).is_ok()
    }
}

impl Drop for VideoPipeline {
    fn drop(&mut self) {
        log::warn!("VideoPipeline dropping - shutting down threads");

        // Send shutdown signals
        let _ = self.command_tx.send(WorkerCommand::Shutdown);
        let _ = self.dropped_tx.send(DroppedFrameCommand::Shutdown);
        let _ = self.shutdown_tx.send(());

        // Give threads a moment to shutdown gracefully
        std::thread::sleep(Duration::from_millis(100));
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
        for i in 0..10 {
            let result = video_pipeline
                .get_result_timeout(Duration::from_secs(5))
                .unwrap();
            println!(
                "Frame {}: {} detections, extrapolated: {}",
                result.frame_id,
                result.detections.len(),
                result.is_extrapolated
            );

            // Test shutdown on the 5th frame
            if i == 4 {
                println!("Testing shutdown...");
                video_pipeline.shutdown();
            }
        }
    }
}
