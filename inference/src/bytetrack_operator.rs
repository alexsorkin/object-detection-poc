use crate::frame_pipeline::TileDetection;
use crate::tracking::TrackingCommand;
use crate::tracking_utils::{tile_detection_to_tracker_format, tracker_output_to_tile_detection};
use ioutrack::ByteMultiTracker;

/// ByteTrack configuration  
#[derive(Clone, Debug)]
pub struct ByteTrackConfig {
    pub max_age: u32,
    pub min_hits: u32,
    pub iou_threshold: f32,
    pub init_tracker_min_score: f32,
    pub high_score_threshold: f32,
    pub low_score_threshold: f32,
    pub measurement_noise: [f32; 4],
    pub process_noise: [f32; 7],
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            max_age: 2,
            min_hits: 1, // Allow immediate track creation (lowered from 3)
            iou_threshold: 0.3,
            init_tracker_min_score: 0.3, // Match RT-DETR confidence levels (~30-40%)
            high_score_threshold: 0.5,
            low_score_threshold: 0.1,
            measurement_noise: [1.0, 1.0, 10.0, 10.0],
            process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        }
    }
}
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;

/// ByteTrack Operator - manages tracking state with async command processing
pub struct ByteTrackOperator {
    command_tx: Sender<TrackingCommand>,
    #[allow(dead_code)] // Used by background worker threads
    tracker: Arc<Mutex<ByteMultiTracker>>,
    last_predictions: Arc<Mutex<Vec<TileDetection>>>,
    _worker_handle: Option<thread::JoinHandle<()>>,
    _runtime: Runtime,
    shutdown_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

static BYTETRACK_INSTANCE: OnceLock<Arc<Mutex<ByteTrackOperator>>> = OnceLock::new();

impl ByteTrackOperator {
    /// Initialize the global operator instance
    pub fn init(tracker_config: ByteTrackConfig) -> Arc<Mutex<Self>> {
        BYTETRACK_INSTANCE
            .get_or_init(|| {
                let tracker = Arc::new(Mutex::new(ByteMultiTracker::new(
                    tracker_config.max_age,
                    tracker_config.min_hits,
                    tracker_config.iou_threshold,
                    tracker_config.init_tracker_min_score,
                    tracker_config.high_score_threshold,
                    tracker_config.low_score_threshold,
                    tracker_config.measurement_noise,
                    tracker_config.process_noise,
                )));
                let last_predictions = Arc::new(Mutex::new(Vec::new()));
                let (command_tx, command_rx) = channel::<TrackingCommand>();

                // Create tokio runtime for maintenance thread
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .thread_name("bytetrack-maintenance")
                    .enable_all()
                    .build()
                    .expect("Failed to create ByteTrack maintenance runtime");

                // Spawn command processor thread
                let tracker_clone = Arc::clone(&tracker);
                let predictions_clone = Arc::clone(&last_predictions);
                let worker_handle = thread::spawn(move || {
                    Self::command_processor(tracker_clone, predictions_clone, command_rx);
                });

                // Spawn maintenance thread (less critical for ByteTrack)
                let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
                let tracker_clone = Arc::clone(&tracker);

                runtime.spawn(async move {
                    Self::maintenance_loop(tracker_clone, shutdown_rx).await;
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
        BYTETRACK_INSTANCE.get().cloned()
    }

    /// Send update command (non-blocking)
    pub fn send_update(&self, detections: Vec<TileDetection>, dt: f32) -> Result<(), String> {
        self.command_tx
            .send(TrackingCommand::Update { detections, dt })
            .map_err(|e| format!("Failed to send update: {}", e))
    }

    /// Send predict command (non-blocking) - no-op for ByteTrack
    pub fn send_predict(&self, dt: f32) -> Result<(), String> {
        self.command_tx
            .send(TrackingCommand::Predict { dt })
            .map_err(|e| format!("Failed to send predict: {}", e))
    }

    /// Get current predictions (synchronous query, fast)
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        let predictions = self.last_predictions.lock().unwrap();
        predictions.clone()
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

    /// Maintenance loop - runs every 50ms for light maintenance
    async fn maintenance_loop(
        tracker: Arc<Mutex<ByteMultiTracker>>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) {
        log::info!("ByteTrack maintenance thread started (checking every 50ms)");

        let mut interval = tokio::time::interval(Duration::from_millis(50));
        let mut maintenance_cycles = 0_u64;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    maintenance_cycles += 1;

                    // ByteTrack handles track lifecycle internally
                    // Light maintenance if needed
                    let tracker = tracker.lock().unwrap();
                    let track_count = tracker.num_tracklets();

                    // Periodic status log
                    if maintenance_cycles % 200 == 0 {
                        log::info!(
                            "ByteTrack maintenance: {} cycles, {} active tracks",
                            maintenance_cycles,
                            track_count
                        );
                    }
                }
                _ = &mut shutdown_rx => {
                    log::info!("ByteTrack maintenance thread shutting down after {} cycles", maintenance_cycles);
                    break;
                }
            }
        }

        log::info!("ByteTrack maintenance thread stopped");
    }

    /// Command processor thread - subscribes to queue and updates tracker
    fn command_processor(
        tracker: Arc<Mutex<ByteMultiTracker>>,
        last_predictions: Arc<Mutex<Vec<TileDetection>>>,
        command_rx: Receiver<TrackingCommand>,
    ) {
        log::info!("ByteTrack tracker command processor started");
        let mut commands_processed = 0_u64;

        loop {
            match command_rx.recv() {
                Ok(TrackingCommand::Update { detections, dt: _ }) => {
                    // Convert TileDetection to ndarray format [x1, y1, x2, y2, confidence]
                    let detection_array = tile_detection_to_tracker_format(&detections);

                    let mut tracker = tracker.lock().unwrap();
                    match tracker.update(detection_array.view(), false, false) {
                        Ok(tracks) => {
                            // Convert tracked results back to TileDetection format
                            let tracked_detections = tracker_output_to_tile_detection(&tracks);

                            commands_processed += 1;

                            // Store predictions for queries
                            {
                                let mut predictions = last_predictions.lock().unwrap();
                                *predictions = tracked_detections;
                            }
                        }
                        Err(e) => {
                            log::error!("ByteTrack update failed: {:?}", e);
                        }
                    }

                    if commands_processed % 100 == 0 {
                        log::debug!(
                            "ByteTrack processed {} updates, {} active tracks",
                            commands_processed,
                            tracker.num_tracklets()
                        );
                    }
                }
                Ok(TrackingCommand::Predict { .. }) => {
                    // ByteTrack doesn't have explicit prediction step
                    // Just increment command counter
                    commands_processed += 1;
                }
                Ok(TrackingCommand::Shutdown) => {
                    log::info!(
                        "ByteTrack tracker shutting down after {} commands",
                        commands_processed
                    );
                    break;
                }
                Err(_) => {
                    log::warn!("ByteTrack command channel disconnected");
                    break;
                }
            }
        }

        log::info!("ByteTrack tracker command processor stopped");
    }
}

impl Drop for ByteTrackOperator {
    fn drop(&mut self) {
        self.shutdown();

        // Wait for command processor thread
        if let Some(handle) = self._worker_handle.take() {
            let _ = handle.join();
        }

        // Runtime will be dropped automatically, cleaning up maintenance thread
    }
}
