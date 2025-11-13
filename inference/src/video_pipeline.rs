use crate::frame_pipeline::{DetectionPipeline, PipelineOutput, TileDetection};
/// Video processing pipeline with async detection and temporal tracking
///
/// Handles real-time video streams with:
/// - Async frame detection with backpressure
/// - Multiple tracking algorithms (Kalman filter or ByteTrack)
/// - Extrapolation when detection latency is high
/// - Pre-allocated buffers for optimal runtime performance
use crate::tracking::{TrackingConfig, UnifiedTracker};
use crossbeam::channel::{bounded, unbounded, Receiver, Sender};

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Frame with metadata - zero-copy data sharing
#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Arc<[u8]>, // Zero-copy frame data using Arc
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

/// Internal tracker commands
enum TrackerCommand {
    ProcessOutput(u64, Instant, PipelineOutput),
    AdvanceTracks,
    Shutdown,
}

/// Video pipeline configuration
pub struct VideoPipelineConfig {
    /// Tracking configuration (Kalman or ByteTrack)
    pub tracking_config: TrackingConfig,
    /// Frame buffer size (number of frames to queue)
    pub buffer_size: usize,
}

impl Default for VideoPipelineConfig {
    fn default() -> Self {
        Self {
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
    tracker_tx: Sender<TrackerCommand>,
    result_rx: Receiver<FrameResult>,
    tracker: Arc<UnifiedTracker>, // Shared reference to tracker for direct access
    _worker_handle: thread::JoinHandle<()>,
    _tracker_handle: thread::JoinHandle<()>,
}

impl VideoPipeline {
    /// Create new video processing pipeline
    pub fn new(detection_pipeline: Arc<DetectionPipeline>, config: VideoPipelineConfig) -> Self {
        let (command_tx, command_rx) = bounded::<WorkerCommand>(config.buffer_size);
        let (tracker_tx, tracker_rx) = unbounded::<TrackerCommand>();

        let (result_tx, result_rx) = unbounded::<FrameResult>();

        let tracker = Arc::new(UnifiedTracker::new(config.tracking_config));
        let detector = Arc::clone(&detection_pipeline);

        // Start unified tracker loop thread - handles both detection results and advance tracks
        let tracker_clone = Arc::clone(&tracker);
        let tracker_handle = thread::spawn(move || {
            Self::tracker_loop(tracker_rx, result_tx, tracker_clone);
        });

        let tracker_tx_worker = tracker_tx.clone();
        let worker_handle = thread::spawn(move || {
            Self::worker_loop(command_rx, detector, tracker_tx_worker);
        });

        Self {
            command_tx,
            tracker_tx,
            result_rx,
            tracker,
            _worker_handle: worker_handle,
            _tracker_handle: tracker_handle,
        }
    }

    /// Unified tracker loop: handles both detection outputs and advance track commands
    fn tracker_loop(
        tracker_rx: Receiver<TrackerCommand>,
        result_tx: Sender<FrameResult>,
        tracker: Arc<UnifiedTracker>,
    ) {
        log::info!(
            "Unified tracker loop started with {} tracker",
            tracker.method()
        );
        let mut frames_extrapolated = 0_u64;
        let mut last_frame_timestamp = Instant::now();

        // Pre-allocated buffers for better runtime performance
        let _empty_detections_buffer = Vec::<TileDetection>::new();

        loop {
            match tracker_rx.recv() {
                Ok(command) => {
                    match command {
                        TrackerCommand::ProcessOutput(frame_id, timestamp, pipeline_output) => {
                            // Calculate dt from frame timestamp to last processed frame
                            let dt = timestamp
                                .duration_since(last_frame_timestamp)
                                .as_secs_f32()
                                .max(0.001) // Minimum 1ms to avoid zero dt
                                .min(0.2); // Maximum 200ms to avoid huge jumps

                            log::debug!(
                                "Tracker update: frame_id={}, dt={:.3}s, {} detections",
                                frame_id,
                                dt,
                                pipeline_output.detections.len()
                            );

                            // Send update command to async tracker
                            if let Err(e) =
                                tracker.send_update(pipeline_output.detections.clone(), dt)
                            {
                                log::error!("Failed to send update to tracker: {}", e);
                            }

                            // Get current cached predictions from tracker
                            let tracked_detections = tracker.get_predictions();

                            // Create frame result with extracted diagnostics
                            let result = FrameResult {
                                frame_id,
                                timestamp,
                                detections: (*tracked_detections).clone(), // Clone only when creating FrameResult
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

                            last_frame_timestamp = timestamp;
                        }
                        TrackerCommand::AdvanceTracks => {
                            let now = Instant::now();

                            let dt = now
                                .duration_since(last_frame_timestamp)
                                .as_secs_f32()
                                .max(0.001) // Minimum 1ms to avoid zero dt
                                .min(0.2); // Maximum 200ms to avoid huge jumps

                            // Send predict command to async tracker
                            if let Err(e) = tracker.send_predict(dt) {
                                log::error!("Failed to send predict to tracker: {}", e);
                            }

                            // Get current cached predictions from tracker
                            let tracked_predictions = tracker.get_predictions();

                            log::debug!(
                                "Advanced tracks: {} predictions, dt={:.3}s",
                                tracked_predictions.len(),
                                dt
                            );

                            // CREATE A RESULT WITH THE KALMAN/BYTETRACK PREDICTIONS!
                            let result = FrameResult {
                                frame_id: frames_extrapolated, // Use extrapolation count as frame ID
                                timestamp: Instant::now(),
                                detections: (*tracked_predictions).clone(), // Clone only when creating FrameResult
                                is_extrapolated: true,                      // Mark as extrapolated
                                processing_time_ms: 0.0, // No detection processing
                                latency_ms: 0.0,         // Instant extrapolation
                                duplicates_removed: 0,
                                nested_removed: 0,
                            };

                            // SEND THE PREDICTED TRACKS TO THE RESULT CHANNEL
                            let num_predictions = result.detections.len();
                            if let Err(e) = result_tx.try_send(result) {
                                log::warn!("Failed to send extrapolated result: {}", e);
                            } else {
                                log::debug!(
                                    "Sent {} predicted tracks as extrapolated result",
                                    num_predictions
                                );
                            }

                            //log::warn!("Advancing trackers for dropped frame (pipeline busy)");

                            frames_extrapolated += 1;

                            // Log statistics every 50 dropped frames
                            if frames_extrapolated % 50 == 0 {
                                log::debug!("Dropped frames handled: {}", frames_extrapolated);
                            }
                        }
                        TrackerCommand::Shutdown => {
                            log::debug!("Tracker loop received shutdown command");
                            break;
                        }
                    }
                }
                Err(_) => {
                    log::info!("Tracker command channel closed");
                    break;
                }
            }
        }

        log::info!(
            "UnifiedTracker: tracker loop stopped (total extrapolated: {})",
            frames_extrapolated
        );
    }

    /// Worker thread: handles commands and triggers detection
    fn worker_loop(
        command_rx: Receiver<WorkerCommand>,
        detector: Arc<DetectionPipeline>,
        tracker_tx: Sender<TrackerCommand>,
    ) {
        let mut frames_processed = 0_u64;

        log::info!("Video pipeline worker started");

        loop {
            match command_rx.recv() {
                Ok(command) => {
                    match command {
                        WorkerCommand::ProcessFrame(frame) => {
                            let now = Instant::now();
                            let frame_id = frame.sequence;
                            let timestamp = now;

                            // Zero-copy: use Arc data efficiently instead of cloning
                            let width = frame.width;
                            let height = frame.height;

                            // Convert Arc<[u8]> to Vec<u8> only when needed for image processing
                            let img_data = frame.data.as_ref().to_vec();

                            // Create an RgbImage from raw data (assuming RGB format)
                            let image = match image::RgbImage::from_raw(width, height, img_data) {
                                Some(img) => img,
                                None => {
                                    log::error!("Failed to create image from frame data");
                                    continue;
                                }
                            };

                            // Trigger detection with callback that sends to tracker loop
                            let tracker_tx_clone = tracker_tx.clone();
                            detector.process_with_callback(
                                &image,
                                move |output: &PipelineOutput| {
                                    // Send frame context and pipeline output to tracker loop
                                    if let Err(e) =
                                        tracker_tx_clone.try_send(TrackerCommand::ProcessOutput(
                                            frame_id,
                                            timestamp,
                                            output.clone(),
                                        ))
                                    {
                                        log::warn!("Failed to send detection output: {}", e);
                                    }
                                },
                            );

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
                            log::debug!("Worker loop received shutdown command");
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

        log::info!("Video processor pipeline stopped");
    }

    /// Submit frame for processing (non-blocking, returns false if buffer full)
    pub fn submit_frame(&self, frame: Frame) -> bool {
        self.command_tx
            .try_send(WorkerCommand::ProcessFrame(frame))
            .is_ok()
    }

    /// Get next result (non-blocking)
    pub fn get_result(&self) -> Option<FrameResult> {
        self.result_rx.try_recv().ok()
    }

    /// Get current predictions directly from tracker cache (synchronous)
    /// Returns immediately with cached tracker predictions
    pub fn get_predictions(&self) -> Arc<Vec<TileDetection>> {
        // Get predictions directly from tracker cache
        // This is always synchronous and returns immediately
        self.tracker.get_predictions()
    }

    /// Advance tracker predictions for dropped frames (async, non-blocking)
    /// This advances the tracker's motion models without processing a frame
    pub fn advance_tracks(&self) -> bool {
        // Send a command to advance tracker predictions to the unified tracker loop
        self.tracker_tx
            .try_send(TrackerCommand::AdvanceTracks)
            .is_ok()
    }

    /// Shutdown the video pipeline gracefully
    /// This will stop all worker threads and close all channels
    pub fn shutdown(&self) {
        log::debug!("Initiating video pipeline shutdown");

        // Send shutdown command to worker
        let _ = self.command_tx.send(WorkerCommand::Shutdown);

        // Send shutdown command to tracker loop
        let _ = self.tracker_tx.send(TrackerCommand::Shutdown);
    }
}

impl Drop for VideoPipeline {
    fn drop(&mut self) {
        log::warn!("VideoPipeline dropping - shutting down threads");

        // Send shutdown signals using poison pill pattern
        let _ = self.command_tx.send(WorkerCommand::Shutdown);
        let _ = self.tracker_tx.send(TrackerCommand::Shutdown);

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
                data: Arc::from(vec![0u8; 640 * 480 * 3].into_boxed_slice()), // Zero-copy Arc data
                width: 640,
                height: 480,
                sequence: i,
            };
            assert!(video_pipeline.submit_frame(frame), "Failed to submit frame");
        }

        // Get results
        for i in 0..10 {
            let result = video_pipeline.get_result().expect("Should have result");
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
