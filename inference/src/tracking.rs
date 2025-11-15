/// Tracking method selection for video pipeline
use crate::frame_pipeline::TileDetection;
use crate::tracking_utils::{
    calculate_iou, tile_detection_to_tracker_format, tracker_output_to_tile_detection, BoundingBox,
};
use crossbeam::channel::{self, Receiver, Sender};
use ioutrack::{ByteMultiTracker, HungarianSolver, KalmanMultiTracker, MultiObjectTracker};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
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
    last_predictions: Arc<RwLock<Arc<Vec<TileDetection>>>>,
    /// Background worker thread handle
    _worker_handle: Option<thread::JoinHandle<()>>,
    /// Tokio runtime for maintenance tasks
    _runtime: Runtime,
    /// Shutdown signal for cleanup
    shutdown_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    /// Maintenance period in milliseconds
    maintenance_period_ms: u64,
}

/// Metadata for each track
#[derive(Clone, Debug)]
struct TrackMetadata {
    class_id: u32,
    class_name: Arc<str>,
    last_confidence: f32,
}

impl std::fmt::Debug for UnifiedTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedTracker")
            .field("method", &self.method)
            .field(
                "predictions_cached",
                &self.last_predictions.read().unwrap().len(),
            )
            .field("maintenance_period_ms", &self.maintenance_period_ms)
            .finish_non_exhaustive()
    }
}

impl UnifiedTracker {
    /// Create new tracker from config with async processing
    pub fn new(config: TrackingConfig) -> Self {
        let (command_tx, command_rx) = channel::unbounded::<TrackingCommand>();
        let last_predictions = Arc::new(RwLock::new(Arc::new(Vec::new())));

        // Extract maintenance period from config
        let maintenance_period_ms = match &config {
            TrackingConfig::Kalman(cfg) => cfg.maintenance_period_ms,
            TrackingConfig::ByteTrack(cfg) => cfg.maintenance_period_ms,
        };

        // Create the appropriate tracker based on config
        let (tracker, method): (Box<dyn MultiObjectTracker>, TrackingMethod) = match &config {
            TrackingConfig::Kalman(kalman_config) => {
                log::info!("Creating KalmanMultiTracker with config: max_age={}, min_hits={}, iou_threshold={:.3}, init_tracker_min_score={:.3}, maintenance_period={}ms", 
                    kalman_config.max_age, kalman_config.min_hits, kalman_config.iou_threshold, kalman_config.init_tracker_min_score, kalman_config.maintenance_period_ms);
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
                log::info!("Creating ByteMultiTracker with config: max_age={}, min_hits={}, iou_threshold={:.3}, init_tracker_min_score={:.3}, maintenance_period={}ms", 
                    bytetrack_config.max_age, bytetrack_config.min_hits, bytetrack_config.iou_threshold, bytetrack_config.init_tracker_min_score, bytetrack_config.maintenance_period_ms);
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
        let command_tx_for_maintenance = command_tx.clone();
        runtime.spawn(async move {
            Self::maintenance_loop(
                &command_tx_for_maintenance,
                shutdown_rx,
                maintenance_period_ms,
            )
            .await;
        });

        Self {
            method,
            command_tx,
            last_predictions,
            _worker_handle: Some(worker_handle),
            _runtime: runtime,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            maintenance_period_ms,
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
        Arc::clone(&*self.last_predictions.read().unwrap())
    }

    /// Synchronous update with immediate results
    /// Sends update command and waits briefly for processing, then returns current predictions
    pub fn update_sync(
        &self,
        detections: Vec<TileDetection>,
        dt: f32,
    ) -> Result<Arc<Vec<TileDetection>>, String> {
        // Send the update command
        self.send_update(detections, dt)?;

        // Give the async processor a moment to process the update
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Return the updated predictions
        Ok(self.get_predictions())
    }

    /// Synchronous update with Arc (zero-copy)
    /// Sends update command and waits briefly for processing, then returns current predictions
    pub fn update_sync_arc(
        &self,
        detections: Arc<Vec<TileDetection>>,
        dt: f32,
    ) -> Result<Arc<Vec<TileDetection>>, String> {
        // Send the update command (zero-copy)
        self.send_update_arc(detections, dt)?;

        // Give the async processor a moment to process the update
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Return the updated predictions
        Ok(self.get_predictions())
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
        self.last_predictions.read().unwrap().len()
    }

    /// Command processor thread - handles tracker updates in background
    fn command_processor(
        tracker: Arc<Mutex<Box<dyn MultiObjectTracker>>>,
        last_predictions: Arc<RwLock<Arc<Vec<TileDetection>>>>,
        command_rx: Receiver<TrackingCommand>,
    ) {
        log::debug!("UnifiedTracker: command processor started");
        let mut commands_processed = 0_u64;
        let mut track_metadata: HashMap<u32, TrackMetadata> = HashMap::with_capacity(64);
        let mut valid_predictions = Vec::with_capacity(64); // Pre-allocate buffer

        loop {
            let mut predictions_to_cache: Option<Arc<Vec<TileDetection>>> = None;

            match command_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(TrackingCommand::Update { detections, dt }) => {
                    log::debug!(
                        "UnifiedTracker: executing update command with {} detections, dt={:.3}s",
                        detections.len(),
                        dt
                    );

                    // Convert TileDetection to tracker format
                    let detection_array = tile_detection_to_tracker_format(&detections);
                    log::debug!(
                        "UnifiedTracker: converted to detection array shape: {:?}",
                        detection_array.dim()
                    );

                    let mut tracker = tracker.lock().unwrap();
                    match tracker.update(detection_array.view(), true, false) {
                        Ok(tracks) => {
                            log::debug!(
                                "UnifiedTracker: update produced {} tracks",
                                tracks.nrows()
                            );

                            // Convert back to TileDetection format
                            let mut tracked_detections = tracker_output_to_tile_detection(&tracks);
                            let num_tracklets = tracker.num_tracklets();
                            // Release tracker lock BEFORE metadata update to reduce contention
                            drop(tracker);

                            // Debug: Log track positions before metadata update
                            for detection in &tracked_detections {
                                if let Some(track_id) = detection.track_id {
                                    log::trace!(
                                        "Track {} update: pos=({:.1}, {:.1}) size=({:.1}x{:.1}) before metadata",
                                        track_id, detection.x, detection.y, detection.w, detection.h
                                    );
                                }
                            }

                            // Update metadata for tracked detections
                            Self::update_track_metadata(
                                &detections,
                                &mut tracked_detections,
                                &mut track_metadata,
                            );

                            commands_processed += 1;

                            // Prepare predictions for batched write
                            predictions_to_cache = Some(Arc::new(tracked_detections));

                            if commands_processed % 100 == 0 {
                                log::debug!(
                                    "UnifiedTracker processed {} updates, {} active tracks",
                                    commands_processed,
                                    num_tracklets
                                );
                            }
                        }
                        Err(e) => {
                            log::error!("UnifiedTracker update failed: {:?}", e);
                        }
                    }
                }
                Ok(TrackingCommand::Predict { dt }) => {
                    log::debug!(
                        "UnifiedTracker: executing predict command with dt={:.3}s",
                        dt
                    );

                    // Get predictions from current tracker state
                    let mut tracker = tracker.lock().unwrap();
                    let empty_detections = ndarray::Array2::zeros((0, 5));

                    match tracker.update(empty_detections.view(), true, false) {
                        Ok(tracks) => {
                            log::debug!("UnifiedTracker: predict got {} tracks", tracks.nrows());
                            let predictions = tracker_output_to_tile_detection(&tracks);
                            // Release tracker lock BEFORE metadata application
                            drop(tracker);

                            // Debug: Log prediction results before metadata application
                            for prediction in &predictions {
                                if let Some(track_id) = prediction.track_id {
                                    log::trace!(
                                        "Track {} predict: pos=({:.1}, {:.1}) size=({:.1}x{:.1}) before metadata",
                                        track_id, prediction.x, prediction.y, prediction.w, prediction.h
                                    );
                                }
                            }

                            // Apply metadata to predictions and filter out tracks without metadata
                            valid_predictions.clear(); // Reuse allocation

                            for mut prediction in predictions {
                                if let Some(track_id) = prediction.track_id {
                                    if let Some(metadata) = track_metadata.get(&track_id) {
                                        prediction.class_id = metadata.class_id;
                                        prediction.class_name = Arc::clone(&metadata.class_name);
                                        prediction.confidence = metadata.last_confidence;

                                        log::trace!(
                                            "Track {} predict: class_id={} '{}' confidence={:.2}",
                                            track_id,
                                            metadata.class_id,
                                            metadata.class_name,
                                            metadata.last_confidence
                                        );

                                        valid_predictions.push(prediction);
                                    } else {
                                        // Track exists but has no metadata yet (newly created or never matched)
                                        // Skip this track - it hasn't been validated by actual detections
                                        log::trace!(
                                            "Track {} predict: skipping - no metadata found",
                                            track_id
                                        );
                                    }
                                }
                            }

                            // Prepare predictions for batched write (move, don't clone)
                            predictions_to_cache =
                                Some(Arc::new(std::mem::take(&mut valid_predictions)));
                        }
                        Err(e) => {
                            log::error!("UnifiedTracker: prediction failed: {:?}", e);
                        }
                    }

                    commands_processed += 1;
                }
                Ok(TrackingCommand::Shutdown) => {
                    log::debug!(
                        "UnifiedTracker: shutting down after {} commands",
                        commands_processed
                    );
                    break;
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    // Periodic maintenance: clean up stale track metadata
                    // For now, just continue - proper cleanup would require tracker API changes
                    continue;
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    log::warn!("UnifiedTracker command channel disconnected");
                    break;
                }
            }

            // Batched write: update predictions cache once per iteration
            if let Some(predictions) = predictions_to_cache {
                let mut cached = last_predictions.write().unwrap();
                *cached = predictions;
            }
        }

        log::debug!("UnifiedTracker: command processor stopped");
    }

    /// Maintenance loop - periodic cleanup and logging
    async fn maintenance_loop(
        command_tx: &Sender<TrackingCommand>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
        maintenance_period_ms: u64,
    ) {
        log::debug!(
            "UnifiedTracker: maintenance thread started with period {}ms",
            maintenance_period_ms
        );

        let mut interval = tokio::time::interval(Duration::from_millis(maintenance_period_ms));
        let mut maintenance_cycles = 0_u64;
        let mut last_frame_timestamp = std::time::Instant::now();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    maintenance_cycles += 1;

                    let now = std::time::Instant::now();

                    let dt = now
                        .duration_since(last_frame_timestamp)
                        .as_secs_f32()
                        .max(0.001) // Minimum 1ms to avoid zero dt
                        .min(0.2); // Maximum 200ms to avoid huge jumps

                    // Advance tracker predictions forward in time (predict step with no new detections)
                    // This updates ByteTrack/Kalman filter states and ages out stale tracks
                    let _ = command_tx.send(TrackingCommand::Predict { dt }).is_ok();

                    if maintenance_cycles % 100 == 0 {
                        log::debug!(
                            "UnifiedTracker: maintenance: {} cycles",
                            maintenance_cycles
                        );
                    }
                    last_frame_timestamp = std::time::Instant::now();
                }
                _ = &mut shutdown_rx => {
                    break;
                }
            }
        }

        log::debug!("UnifiedTracker: maintenance thread stopped");
    }

    /// Update track metadata for tracked detections using Hungarian algorithm
    fn update_track_metadata(
        detections: &Arc<Vec<TileDetection>>,
        tracked_detections: &mut [TileDetection],
        track_metadata: &mut HashMap<u32, TrackMetadata>,
    ) {
        if detections.is_empty() || tracked_detections.is_empty() {
            return;
        }

        // Create IoU matrix: rows = detections, cols = tracked_detections
        let num_detections = detections.len();
        let num_tracks = tracked_detections.len();
        let mut iou_matrix = ndarray::Array2::zeros((num_detections, num_tracks));

        // Calculate IoU between all detection-track pairs
        for (det_idx, detection) in detections.iter().enumerate() {
            let det_box = BoundingBox {
                x1: detection.x,
                y1: detection.y,
                x2: detection.x + detection.w,
                y2: detection.y + detection.h,
            };

            for (track_idx, tracked_det) in tracked_detections.iter().enumerate() {
                let track_box = BoundingBox {
                    x1: tracked_det.x,
                    y1: tracked_det.y,
                    x2: tracked_det.x + tracked_det.w,
                    y2: tracked_det.y + tracked_det.h,
                };

                let iou = calculate_iou(&det_box, &track_box);
                iou_matrix[[det_idx, track_idx]] = iou;
            }
        }

        // Use Hungarian algorithm for optimal assignment
        let assignment_result = HungarianSolver::solve_iou(iou_matrix.view(), 0.3);

        log::trace!(
            "update_track_metadata: {} detections, {} tracks, {} assignments",
            num_detections,
            num_tracks,
            assignment_result.assignments.len()
        );

        // Apply metadata for valid assignments
        for (det_idx, track_idx) in assignment_result.assignments {
            if let (Some(detection), Some(tracked_det)) = (
                detections.get(det_idx),
                tracked_detections.get_mut(track_idx),
            ) {
                if let Some(track_id) = tracked_det.track_id {
                    let class_id = detection.class_id;
                    let class_name = Arc::clone(&detection.class_name);
                    let confidence = detection.confidence;

                    track_metadata.insert(
                        track_id,
                        TrackMetadata {
                            class_id,
                            class_name: Arc::clone(&class_name),
                            last_confidence: confidence,
                        },
                    );

                    // Apply metadata to tracked detection
                    tracked_det.class_id = class_id;
                    tracked_det.class_name = Arc::clone(&class_name);
                    tracked_det.confidence = confidence;

                    log::trace!(
                        "Track {} assigned to detection: class_id={} '{}' conf={:.2}",
                        track_id,
                        class_id,
                        class_name,
                        confidence
                    );
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
