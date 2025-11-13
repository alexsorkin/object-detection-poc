/// Tracking method selection for video pipeline
use crate::frame_pipeline::TileDetection;
use crate::tracking_utils::{
    calculate_iou, tile_detection_to_tracker_format, tracker_output_to_tile_detection, BoundingBox,
};
use crossbeam::channel::{self, Receiver, Sender};
use ioutrack::{ByteMultiTracker, KalmanMultiTracker, MultiObjectTracker};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Commands that can be sent to any tracking operator
#[derive(Debug)]
pub enum TrackingCommand {
    /// Update tracker with new detections
    Update {
        detections: Arc<Vec<TileDetection>>,
        dt: f32,
    },
    /// Predict forward in time without detections
    Predict { dt: f32 },
    /// Shutdown the tracker thread
    Shutdown,
}

/// Available tracking algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackingMethod {
    /// Kalman filter-based tracking with Hungarian data association
    Kalman,
    /// ByteTrack algorithm (more modern, handles occlusions better)
    ByteTrack,
}

impl Default for TrackingMethod {
    fn default() -> Self {
        Self::Kalman
    }
}

impl std::fmt::Display for TrackingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kalman => write!(f, "Kalman Filter"),
            Self::ByteTrack => write!(f, "ByteTrack"),
        }
    }
}

/// Configuration for tracking methods
#[derive(Debug, Clone)]
pub enum TrackingConfig {
    Kalman(crate::tracking_types::KalmanConfig),
    ByteTrack(crate::tracking_types::ByteTrackConfig),
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self::Kalman(crate::tracking_types::KalmanConfig::default())
    }
}

impl TrackingConfig {
    /// Get the tracking method from config
    pub fn method(&self) -> TrackingMethod {
        match self {
            Self::Kalman(_) => TrackingMethod::Kalman,
            Self::ByteTrack(_) => TrackingMethod::ByteTrack,
        }
    }
}

/// Unified tracker interface that can use any tracking algorithm with async processing
pub struct UnifiedTracker {
    /// Tracking method being used
    method: TrackingMethod,
    /// Command channel for async operations
    command_tx: Sender<TrackingCommand>,
    /// Cached predictions for fast synchronous access
    last_predictions: Arc<Mutex<Arc<Vec<TileDetection>>>>,
    /// Background worker thread handle
    _worker_handle: Option<thread::JoinHandle<()>>,
    /// Tokio runtime for maintenance tasks
    _runtime: Runtime,
    /// Shutdown signal for cleanup
    shutdown_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

/// Metadata for each track
#[derive(Clone, Debug)]
struct TrackMetadata {
    class_id: u32,
    class_name: Arc<String>,
    last_confidence: f32,
}

impl std::fmt::Debug for UnifiedTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedTracker")
            .field("method", &self.method)
            .field(
                "predictions_cached",
                &self.last_predictions.lock().unwrap().len(),
            )
            .finish_non_exhaustive()
    }
}

impl UnifiedTracker {
    /// Create new tracker from config with async processing
    pub fn new(config: TrackingConfig) -> Self {
        let (command_tx, command_rx) = channel::unbounded::<TrackingCommand>();
        let last_predictions = Arc::new(Mutex::new(Arc::new(Vec::new())));

        // Create the appropriate tracker based on config
        let (tracker, method): (Box<dyn MultiObjectTracker>, TrackingMethod) = match &config {
            TrackingConfig::Kalman(kalman_config) => {
                log::info!("Creating KalmanMultiTracker with config: max_age={}, min_hits={}, iou_threshold={:.3}, init_tracker_min_score={:.3}", 
                    kalman_config.max_age, kalman_config.min_hits, kalman_config.iou_threshold, kalman_config.init_tracker_min_score);
                let tracker = KalmanMultiTracker::new(
                    kalman_config.max_age,
                    kalman_config.min_hits,
                    kalman_config.iou_threshold,
                    kalman_config.init_tracker_min_score,
                    kalman_config.measurement_noise,
                    kalman_config.process_noise,
                );
                (Box::new(tracker), TrackingMethod::Kalman)
            }
            TrackingConfig::ByteTrack(bytetrack_config) => {
                log::info!("Creating ByteMultiTracker with config: max_age={}, min_hits={}, iou_threshold={:.3}, init_tracker_min_score={:.3}", 
                    bytetrack_config.max_age, bytetrack_config.min_hits, bytetrack_config.iou_threshold, bytetrack_config.init_tracker_min_score);
                let tracker = ByteMultiTracker::new(
                    bytetrack_config.max_age,
                    bytetrack_config.min_hits,
                    bytetrack_config.iou_threshold,
                    bytetrack_config.init_tracker_min_score,
                    bytetrack_config.high_score_threshold,
                    bytetrack_config.low_score_threshold,
                    bytetrack_config.measurement_noise,
                    bytetrack_config.process_noise,
                );
                (Box::new(tracker), TrackingMethod::ByteTrack)
            }
        };

        let tracker_arc = Arc::new(Mutex::new(tracker));
        let predictions_clone = Arc::clone(&last_predictions);

        // Create tokio runtime for maintenance
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("tracker-maintenance")
            .enable_all()
            .build()
            .expect("Failed to create UnifiedTracker maintenance thread");

        // Spawn command processor thread
        let tracker_clone = Arc::clone(&tracker_arc);
        let worker_handle = thread::spawn(move || {
            Self::command_processor(tracker_clone, predictions_clone, command_rx);
        });

        // Spawn maintenance thread
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let tracker_clone = Arc::clone(&tracker_arc);
        let predictions_clone = Arc::clone(&last_predictions);
        runtime.spawn(async move {
            Self::maintenance_loop(tracker_clone, predictions_clone, shutdown_rx).await;
        });

        Self {
            method,
            command_tx,
            last_predictions,
            _worker_handle: Some(worker_handle),
            _runtime: runtime,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
        }
    }

    /// Send update command (non-blocking)
    pub fn send_update(&self, detections: Vec<TileDetection>, dt: f32) -> Result<(), String> {
        self.command_tx
            .send(TrackingCommand::Update {
                detections: Arc::new(detections),
                dt,
            })
            .map_err(|e| format!("Failed to send update: {}", e))
    }

    /// Send update command with pre-wrapped Arc (zero-copy, non-blocking)
    pub fn send_update_arc(
        &self,
        detections: Arc<Vec<TileDetection>>,
        dt: f32,
    ) -> Result<(), String> {
        self.command_tx
            .send(TrackingCommand::Update { detections, dt })
            .map_err(|e| format!("Failed to send update: {}", e))
    }

    /// Send predict command (non-blocking)  
    pub fn send_predict(&self, dt: f32) -> Result<(), String> {
        self.command_tx
            .send(TrackingCommand::Predict { dt })
            .map_err(|e| format!("Failed to send predict: {}", e))
    }

    /// Get cached predictions (synchronous, fast)
    pub fn get_predictions(&self) -> Arc<Vec<TileDetection>> {
        Arc::clone(&*self.last_predictions.lock().unwrap())
    }

    /// Shutdown the tracker gracefully
    pub fn shutdown(&self) {
        // Send shutdown command to worker thread
        let _ = self.command_tx.send(TrackingCommand::Shutdown);

        // Signal maintenance loop to stop
        if let Ok(mut shutdown_tx) = self.shutdown_tx.lock() {
            if let Some(tx) = shutdown_tx.take() {
                let _ = tx.send(());
            }
        }
    }

    /// Get tracking method
    pub fn method(&self) -> TrackingMethod {
        self.method
    }

    /// Get number of active tracks (estimated from cached predictions)
    pub fn num_tracks(&self) -> usize {
        self.last_predictions.lock().unwrap().len()
    }

    /// Command processor thread - handles tracker updates in background
    fn command_processor(
        tracker: Arc<Mutex<Box<dyn MultiObjectTracker>>>,
        last_predictions: Arc<Mutex<Arc<Vec<TileDetection>>>>,
        command_rx: Receiver<TrackingCommand>,
    ) {
        log::info!("UnifiedTracker command processor started");
        let mut commands_processed = 0_u64;
        let mut track_metadata: HashMap<u32, TrackMetadata> = HashMap::with_capacity(64);

        loop {
            match command_rx.recv() {
                Ok(TrackingCommand::Update { detections, dt: _ }) => {
                    // Convert TileDetection to tracker format
                    let detection_array = tile_detection_to_tracker_format(&detections);

                    let mut tracker = tracker.lock().unwrap();
                    match tracker.update(detection_array.view(), true, false) {
                        Ok(tracks) => {
                            // Convert back to TileDetection format
                            let mut tracked_detections = tracker_output_to_tile_detection(&tracks);

                            // Update metadata for tracked detections
                            Self::update_track_metadata(
                                &detections,
                                &mut tracked_detections,
                                &mut track_metadata,
                            );

                            commands_processed += 1;

                            // Store predictions for fast access
                            {
                                let mut predictions = last_predictions.lock().unwrap();
                                *predictions = Arc::new(tracked_detections);
                            }

                            if commands_processed % 100 == 0 {
                                log::debug!(
                                    "UnifiedTracker processed {} updates, {} active tracks",
                                    commands_processed,
                                    tracker.num_tracklets()
                                );
                            }
                        }
                        Err(e) => {
                            log::error!("UnifiedTracker update failed: {:?}", e);
                        }
                    }
                }
                Ok(TrackingCommand::Predict { dt: _ }) => {
                    // Get predictions from current tracker state
                    let mut tracker = tracker.lock().unwrap();
                    let empty_detections = ndarray::Array2::zeros((0, 5));

                    match tracker.update(empty_detections.view(), true, false) {
                        Ok(tracks) => {
                            let mut predictions = tracker_output_to_tile_detection(&tracks);

                            // Apply metadata to predictions
                            for prediction in &mut predictions {
                                if let Some(track_id) = prediction.track_id {
                                    if let Some(metadata) = track_metadata.get(&track_id) {
                                        prediction.class_id = metadata.class_id;
                                        prediction.class_name = (*metadata.class_name).clone();
                                        prediction.confidence = metadata.last_confidence;
                                    }
                                }
                            }

                            // Store predictions
                            {
                                let mut cached_predictions = last_predictions.lock().unwrap();
                                *cached_predictions = Arc::new(predictions);
                            }
                        }
                        Err(e) => {
                            log::error!("UnifiedTracker prediction failed: {:?}", e);
                        }
                    }

                    commands_processed += 1;
                }
                Ok(TrackingCommand::Shutdown) => {
                    log::info!(
                        "UnifiedTracker shutting down after {} commands",
                        commands_processed
                    );
                    break;
                }
                Err(_) => {
                    log::warn!("UnifiedTracker command channel disconnected");
                    break;
                }
            }
        }

        log::info!("UnifiedTracker command processor stopped");
    }

    /// Maintenance loop - periodic cleanup and logging
    async fn maintenance_loop(
        tracker: Arc<Mutex<Box<dyn MultiObjectTracker>>>,
        last_predictions: Arc<Mutex<Arc<Vec<TileDetection>>>>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) {
        log::info!("UnifiedTracker maintenance thread started (updating predictions every 10ms)");

        let mut interval = tokio::time::interval(Duration::from_millis(10));
        let mut maintenance_cycles = 0_u64;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    maintenance_cycles += 1;

                    // Update predictions from current tracker state every iteration
                    let track_count = {
                        let mut tracker = tracker.lock().unwrap();
                        let empty_detections = ndarray::Array2::zeros((0, 5));

                        // Advance tracker predictions forward in time (predict step with no new detections)
                        // This updates internal Kalman filter states and ages out stale tracks
                        match tracker.update(empty_detections.view(), true, false) {
                            Ok(tracks) => {
                                let predictions = tracker_output_to_tile_detection(&tracks);

                                // Update cached predictions with latest tracker state
                                {
                                    let mut cached_predictions = last_predictions.lock().unwrap();
                                    *cached_predictions = Arc::new(predictions);
                                }

                                tracker.num_tracklets() // Return track count
                            }
                            Err(e) => {
                                log::error!("UnifiedTracker maintenance prediction failed: {:?}", e);
                                0
                            }
                        }
                    };                    // Periodic status log
                    if maintenance_cycles % 100 == 0 {
                        log::debug!(
                            "UnifiedTracker: maintenance: {} cycles, {} active tracks",
                            maintenance_cycles,
                            track_count
                        );
                    }
                }
                _ = &mut shutdown_rx => {
                    log::debug!("UnifiedTracker: maintenance received shutdown signal");
                    break;
                }
            }
        }

        log::info!("UnifiedTracker: maintenance thread stopped");
    }
    /// Update track metadata for tracked detections
    fn update_track_metadata(
        detections: &[TileDetection],
        tracked_detections: &mut [TileDetection],
        track_metadata: &mut HashMap<u32, TrackMetadata>,
    ) {
        if detections.is_empty() || tracked_detections.is_empty() {
            return;
        }

        // Simple IoU matching for metadata assignment
        for tracked_det in tracked_detections.iter_mut() {
            if let Some(track_id) = tracked_det.track_id {
                // Find best matching detection
                let mut best_match = None;
                let mut best_iou = 0.3; // Minimum IoU threshold

                for (det_idx, detection) in detections.iter().enumerate() {
                    let det_box = BoundingBox {
                        x1: detection.x,
                        y1: detection.y,
                        x2: detection.x + detection.w,
                        y2: detection.y + detection.h,
                    };

                    let track_box = BoundingBox {
                        x1: tracked_det.x,
                        y1: tracked_det.y,
                        x2: tracked_det.x + tracked_det.w,
                        y2: tracked_det.y + tracked_det.h,
                    };

                    let iou = calculate_iou(&det_box, &track_box);
                    if iou > best_iou {
                        best_iou = iou;
                        best_match = Some(det_idx);
                    }
                }

                // Update metadata if we found a good match
                if let Some(det_idx) = best_match {
                    if let Some(detection) = detections.get(det_idx) {
                        let metadata = TrackMetadata {
                            class_id: detection.class_id,
                            class_name: Arc::new(detection.class_name.clone()),
                            last_confidence: detection.confidence,
                        };

                        track_metadata.insert(track_id, metadata.clone());

                        // Apply metadata to tracked detection
                        tracked_det.class_id = metadata.class_id;
                        tracked_det.class_name = (*metadata.class_name).clone();
                        tracked_det.confidence = metadata.last_confidence;
                    }
                }
            }
        }
    }
}

impl Drop for UnifiedTracker {
    fn drop(&mut self) {
        log::debug!("UnifiedTracker dropping - shutting down background threads");

        // Send shutdown commands
        let _ = self.command_tx.send(TrackingCommand::Shutdown);

        if let Ok(mut shutdown_tx) = self.shutdown_tx.lock() {
            if let Some(tx) = shutdown_tx.take() {
                let _ = tx.send(());
            }
        }

        // Give threads a moment to shutdown gracefully
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
