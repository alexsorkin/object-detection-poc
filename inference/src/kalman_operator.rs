/// Kalman Operator - Singleton tracker with command queue
///
/// Provides a global operator instance with:
/// - Command queue for async updates from detector
/// - Query interface for predictions
/// - Single thread processing updates
use crate::frame_pipeline::TileDetection;
use crate::kalman_tracker::MultiObjectTracker;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

/// Commands that can be sent to the Kalman operator
#[derive(Debug)]
pub enum KalmanCommand {
    /// Update tracker with new detections
    Update {
        detections: Vec<TileDetection>,
        dt: f32,
    },
    /// Predict forward in time without detections
    Predict { dt: f32 },
    /// Shutdown the tracker thread
    Shutdown,
}

/// Kalman Operator - manages tracking state with async command processing
pub struct KalmanOperator {
    command_tx: Sender<KalmanCommand>,
    tracker: Arc<Mutex<MultiObjectTracker>>,
    _worker_handle: Option<thread::JoinHandle<()>>,
}

static KALMAN_INSTANCE: OnceLock<Arc<Mutex<KalmanOperator>>> = OnceLock::new();

impl KalmanOperator {
    /// Initialize the global operator instance
    pub fn init(tracker_config: crate::kalman_tracker::KalmanConfig) -> Arc<Mutex<Self>> {
        KALMAN_INSTANCE
            .get_or_init(|| {
                let tracker = Arc::new(Mutex::new(MultiObjectTracker::new(tracker_config)));
                let (command_tx, command_rx) = channel::<KalmanCommand>();

                let tracker_clone = Arc::clone(&tracker);
                let worker_handle = thread::spawn(move || {
                    Self::command_processor(tracker_clone, command_rx);
                });

                Arc::new(Mutex::new(Self {
                    command_tx,
                    tracker,
                    _worker_handle: Some(worker_handle),
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
            .send(KalmanCommand::Update { detections, dt })
            .map_err(|e| format!("Failed to send update: {}", e))
    }

    /// Send predict command (non-blocking)
    pub fn send_predict(&self, dt: f32) -> Result<(), String> {
        self.command_tx
            .send(KalmanCommand::Predict { dt })
            .map_err(|e| format!("Failed to send predict: {}", e))
    }

    /// Get current predictions (synchronous query, fast)
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        let tracker = self.tracker.lock().unwrap();
        tracker.get_predictions()
    }

    /// Shutdown the tracker thread
    pub fn shutdown(&self) {
        let _ = self.command_tx.send(KalmanCommand::Shutdown);
    }

    /// Command processor thread - subscribes to queue and updates tracker
    fn command_processor(
        tracker: Arc<Mutex<MultiObjectTracker>>,
        command_rx: Receiver<KalmanCommand>,
    ) {
        log::info!("Kalman tracker command processor started");
        let mut commands_processed = 0_u64;

        loop {
            match command_rx.recv() {
                Ok(KalmanCommand::Update { detections, dt }) => {
                    let mut tracker = tracker.lock().unwrap();
                    tracker.update(&detections, dt);
                    commands_processed += 1;

                    if commands_processed % 100 == 0 {
                        log::debug!(
                            "Kalman processed {} updates, {} active tracks",
                            commands_processed,
                            tracker.get_predictions().len()
                        );
                    }
                }
                Ok(KalmanCommand::Predict { dt }) => {
                    let mut tracker = tracker.lock().unwrap();
                    tracker.update(&[], dt); // Predict without measurements
                    commands_processed += 1;
                }
                Ok(KalmanCommand::Shutdown) => {
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
        if let Some(handle) = self._worker_handle.take() {
            let _ = handle.join();
        }
    }
}
