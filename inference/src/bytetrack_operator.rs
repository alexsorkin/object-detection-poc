use crate::bytetrack_tracker::MultiObjectTracker;
/// ByteTrack Operator - Singleton tracker with command queue
///
/// Provides a global operator instance with:
/// - Command queue for async updates from detector
/// - Query interface for predictions  
/// - Single thread processing updates
/// - Maintenance thread for periodic cleanup (less needed than Kalman)
use crate::frame_pipeline::TileDetection;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Commands that can be sent to the ByteTrack operator
#[derive(Debug)]
pub enum ByteTrackCommand {
    /// Update tracker with new detections
    Update {
        detections: Vec<TileDetection>,
        dt: f32,
    },
    /// Predict forward in time without detections (no-op for ByteTrack)
    Predict { dt: f32 },
    /// Shutdown the tracker thread
    Shutdown,
}

/// ByteTrack Operator - manages tracking state with async command processing
pub struct ByteTrackOperator {
    command_tx: Sender<ByteTrackCommand>,
    tracker: Arc<Mutex<MultiObjectTracker>>,
    last_predictions: Arc<Mutex<Vec<TileDetection>>>,
    _worker_handle: Option<thread::JoinHandle<()>>,
    _runtime: Runtime,
    shutdown_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

static BYTETRACK_INSTANCE: OnceLock<Arc<Mutex<ByteTrackOperator>>> = OnceLock::new();

impl ByteTrackOperator {
    /// Initialize the global operator instance
    pub fn init(tracker_config: crate::bytetrack_tracker::ByteTrackConfig) -> Arc<Mutex<Self>> {
        BYTETRACK_INSTANCE
            .get_or_init(|| {
                let tracker = Arc::new(Mutex::new(MultiObjectTracker::new(tracker_config.clone())));
                let last_predictions = Arc::new(Mutex::new(Vec::new()));
                let (command_tx, command_rx) = channel::<ByteTrackCommand>();

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
            .send(ByteTrackCommand::Update { detections, dt })
            .map_err(|e| format!("Failed to send update: {}", e))
    }

    /// Send predict command (non-blocking) - no-op for ByteTrack
    pub fn send_predict(&self, dt: f32) -> Result<(), String> {
        self.command_tx
            .send(ByteTrackCommand::Predict { dt })
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
        let _ = self.command_tx.send(ByteTrackCommand::Shutdown);
    }

    /// Maintenance loop - runs every 50ms for light maintenance
    async fn maintenance_loop(
        tracker: Arc<Mutex<MultiObjectTracker>>,
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
                    let track_count = tracker.num_tracks();

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
        tracker: Arc<Mutex<MultiObjectTracker>>,
        last_predictions: Arc<Mutex<Vec<TileDetection>>>,
        command_rx: Receiver<ByteTrackCommand>,
    ) {
        log::info!("ByteTrack tracker command processor started");
        let mut commands_processed = 0_u64;

        loop {
            match command_rx.recv() {
                Ok(ByteTrackCommand::Update { detections, dt }) => {
                    let mut tracker = tracker.lock().unwrap();
                    let tracked_detections = tracker.update(&detections, dt);
                    commands_processed += 1;

                    // Store predictions for queries
                    {
                        let mut predictions = last_predictions.lock().unwrap();
                        *predictions = tracked_detections;
                    }

                    if commands_processed % 100 == 0 {
                        log::debug!(
                            "ByteTrack processed {} updates, {} active tracks",
                            commands_processed,
                            tracker.num_tracks()
                        );
                    }
                }
                Ok(ByteTrackCommand::Predict { .. }) => {
                    // ByteTrack doesn't have explicit prediction step
                    // Just increment command counter
                    commands_processed += 1;
                }
                Ok(ByteTrackCommand::Shutdown) => {
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
