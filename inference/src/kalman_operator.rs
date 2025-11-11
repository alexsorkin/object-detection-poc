/// Kalman Operator - Singleton tracker with command queue
///
/// Provides a global operator instance with:
/// - Command queue for async updates from detector
/// - Query interface for predictions
/// - Single thread processing updates
/// - Maintenance thread for periodic track cleanup
use crate::frame_pipeline::TileDetection;
use crate::tracking::TrackingCommand;
use crate::tracking_utils::{tile_detection_to_tracker_format, tracker_output_to_tile_detection};
use ioutrack::{KalmanMultiTracker, MultiObjectTracker};
use ndarray::Array2;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Configuration for Kalman filter noise parameters
#[derive(Clone, Debug)]
pub struct KalmanConfig {
    /// Maximum frames to keep a track alive without matching detections
    pub max_age: u32,
    /// Minimum consecutive hits before track is confirmed
    pub min_hits: u32,
    /// IoU threshold for associating detections to tracks
    pub iou_threshold: f32,
    /// Minimum score to create a new tracklet from unmatched detection
    pub init_tracker_min_score: f32,
    /// Measurement noise covariance diagonal
    pub measurement_noise: [f32; 4],
    /// Process noise covariance diagonal  
    pub process_noise: [f32; 7],
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            max_age: 2,
            min_hits: 1, // Allow immediate track creation (lowered from 2)
            iou_threshold: 0.3,
            init_tracker_min_score: 0.3, // Match RT-DETR confidence levels (~30-40%)
            measurement_noise: [1.0, 1.0, 10.0, 10.0],
            process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        }
    }
}

/// Kalman Operator - manages tracking state with async command processing
/// Kalman Operator - Singleton tracker with command queue
pub struct KalmanOperator {
    command_tx: Sender<TrackingCommand>,
    #[allow(dead_code)] // Used by background worker threads
    tracker: Arc<Mutex<KalmanMultiTracker>>,
    last_predictions: Arc<Mutex<Vec<TileDetection>>>, // Store latest tracking results
    _worker_handle: Option<thread::JoinHandle<()>>,
    _runtime: Runtime,
    shutdown_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}
static KALMAN_INSTANCE: OnceLock<Arc<Mutex<KalmanOperator>>> = OnceLock::new();

impl KalmanOperator {
    /// Initialize the global operator instance
    pub fn init(tracker_config: KalmanConfig) -> Arc<Mutex<Self>> {
        KALMAN_INSTANCE
            .get_or_init(|| {
                let tracker = Arc::new(Mutex::new(KalmanMultiTracker::new(
                    tracker_config.max_age,
                    tracker_config.min_hits,
                    tracker_config.iou_threshold,
                    tracker_config.init_tracker_min_score,
                    tracker_config.measurement_noise,
                    tracker_config.process_noise,
                )));
                let last_predictions = Arc::new(Mutex::new(Vec::new()));
                let (command_tx, command_rx) = channel::<TrackingCommand>();

                // Create tokio runtime for maintenance thread
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .thread_name("kalman-maintenance")
                    .enable_all()
                    .build()
                    .expect("Failed to create Kalman maintenance runtime");

                // Spawn command processor thread
                let tracker_clone = Arc::clone(&tracker);
                let predictions_clone = Arc::clone(&last_predictions);
                let worker_handle = thread::spawn(move || {
                    Self::command_processor(tracker_clone, predictions_clone, command_rx);
                });

                // Spawn maintenance thread
                let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
                let tracker_clone = Arc::clone(&tracker);
                let max_age_ms = tracker_config.max_age as u64;

                runtime.spawn(async move {
                    Self::maintenance_loop(tracker_clone, max_age_ms, shutdown_rx).await;
                });

                Arc::new(Mutex::new(Self {
                    command_tx,
                    tracker,
                    last_predictions,
                    _worker_handle: Some(worker_handle),
                    _runtime: runtime,
                    shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
                }))
            })
            .clone()
    }

    /// Get the global singleton instance (must be initialized first)
    pub fn instance() -> Option<Arc<Mutex<Self>>> {
        KALMAN_INSTANCE.get().cloned()
    }

    /// Send update command (non-blocking)
    pub fn send_update(&self, detections: Vec<TileDetection>, dt: f32) -> Result<(), String> {
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

    /// Get current predictions (synchronous query, fast)
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        self.last_predictions.lock().unwrap().clone()
    }

    /// Shutdown the tracker thread
    pub fn shutdown(&self) {
        // Shutdown maintenance thread
        if let Some(tx) = self.shutdown_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }

        // Shutdown command processor
        let _ = self.command_tx.send(TrackingCommand::Shutdown);
    }

    /// Maintenance loop - runs every 20ms to evict stale tracks
    async fn maintenance_loop(
        tracker: Arc<Mutex<KalmanMultiTracker>>,
        max_age_ms: u64,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) {
        log::info!(
            "Kalman maintenance thread started (checking every 20ms, TTL={}ms)",
            max_age_ms
        );

        let mut interval = tokio::time::interval(Duration::from_millis(20));
        let mut maintenance_cycles = 0_u64;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    maintenance_cycles += 1;

                    // Evict stale tracks
                    let mut tracker = tracker.lock().unwrap();
                    let before_count = tracker.num_tracklets();

                    // The unified interface doesn't expose evict_stale_tracks directly
                    // We can trigger cleanup by calling update with empty detection array
                    let empty_detections = Array2::zeros((0, 5));
                    let _ = tracker.update(empty_detections.view(), false, false);

                    let after_count = tracker.num_tracklets();
                    let evicted = before_count - after_count;

                    if evicted > 0 {
                        log::debug!(
                            "🧹 Maintenance cycle {}: evicted {} stale tracks ({} remaining)",
                            maintenance_cycles,
                            evicted,
                            after_count
                        );
                    }

                    // Periodic status log
                    if maintenance_cycles % 500 == 0 {
                        log::info!(
                            "Kalman maintenance: {} cycles, {} active tracks",
                            maintenance_cycles,
                            after_count
                        );
                    }
                }
                _ = &mut shutdown_rx => {
                    log::info!("Kalman maintenance thread shutting down after {} cycles", maintenance_cycles);
                    break;
                }
            }
        }

        log::info!("Kalman maintenance thread stopped");
    }

    /// Command processor thread - subscribes to queue and updates tracker
    fn command_processor(
        tracker: Arc<Mutex<KalmanMultiTracker>>,
        predictions: Arc<Mutex<Vec<TileDetection>>>,
        command_rx: Receiver<TrackingCommand>,
    ) {
        log::info!("Kalman tracker command processor started");
        let mut commands_processed = 0_u64;

        loop {
            match command_rx.recv() {
                Ok(TrackingCommand::Update { detections, dt: _ }) => {
                    // Convert TileDetection to tracker format
                    let detection_array = tile_detection_to_tracker_format(&detections);

                    let mut tracker = tracker.lock().unwrap();
                    match tracker.update(detection_array.view(), false, false) {
                        Ok(tracks) => {
                            // Convert tracking results back to TileDetection format and store
                            let tracked_detections = tracker_output_to_tile_detection(&tracks);
                            *predictions.lock().unwrap() = tracked_detections;

                            log::debug!(
                                "Kalman updated: {} tracks from {} detections",
                                tracks.nrows(),
                                detections.len()
                            );
                        }
                        Err(e) => {
                            log::error!("Kalman update failed: {:?}", e);
                        }
                    }
                    commands_processed += 1;

                    if commands_processed % 100 == 0 {
                        log::debug!(
                            "Kalman processed {} updates, {} active tracks",
                            commands_processed,
                            tracker.num_tracklets()
                        );
                    }
                }
                Ok(TrackingCommand::Predict { dt: _ }) => {
                    let mut tracker = tracker.lock().unwrap();
                    // Update with empty detections for prediction-only step
                    let empty_detections = Array2::zeros((0, 5));
                    match tracker.update(empty_detections.view(), false, false) {
                        Ok(tracks) => {
                            // Store prediction results
                            let predicted_detections = tracker_output_to_tile_detection(&tracks);
                            *predictions.lock().unwrap() = predicted_detections;

                            log::debug!("Kalman predicted: {} tracks", tracks.nrows());
                        }
                        Err(e) => {
                            log::error!("Kalman predict failed: {:?}", e);
                        }
                    }
                    commands_processed += 1;
                }
                Ok(TrackingCommand::Shutdown) => {
                    log::info!(
                        "Kalman tracker shutting down after {} commands",
                        commands_processed
                    );
                    break;
                }
                Err(_) => {
                    log::warn!("Kalman command channel disconnected");
                    break;
                }
            }
        }

        log::info!("Kalman tracker command processor stopped");
    }
}

impl Drop for KalmanOperator {
    fn drop(&mut self) {
        self.shutdown();

        // Wait for command processor thread
        if let Some(handle) = self._worker_handle.take() {
            let _ = handle.join();
        }

        // Runtime will be dropped automatically, cleaning up maintenance thread
    }
}
