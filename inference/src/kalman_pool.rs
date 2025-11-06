/// Dedicated worker pool for Kalman filter predictions
/// This allows fast parallel extrapolation without blocking on slow detection pipeline
use crate::kalman_tracker::MultiObjectTracker;
use crate::pipeline::TileDetection;
use crossbeam::channel::{bounded, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

pub struct KalmanJob {
    pub frame_id: u64,
    pub timestamp: Instant,
    pub dt: f32, // Time delta for prediction
}

pub struct KalmanResult {
    pub frame_id: u64,
    pub timestamp: Instant,
    pub detections: Vec<TileDetection>,
    pub latency_ms: f32,
}

/// Pool of workers for fast Kalman predictions
pub struct KalmanPool {
    job_tx: Sender<KalmanJob>,
    result_rx: Receiver<KalmanResult>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl KalmanPool {
    /// Create a new Kalman worker pool
    pub fn new(num_workers: usize, tracker: Arc<Mutex<MultiObjectTracker>>) -> Self {
        let (job_tx, job_rx) = bounded::<KalmanJob>(num_workers * 2);
        let (result_tx, result_rx) = bounded::<KalmanResult>(num_workers * 2);

        let mut workers = Vec::new();

        for worker_id in 0..num_workers {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let tracker = Arc::clone(&tracker);

            let handle = thread::spawn(move || {
                log::debug!("Kalman worker {} started", worker_id);

                while let Ok(job) = job_rx.recv() {
                    let start = Instant::now();

                    // Lock tracker, get predictions
                    let detections = {
                        let mut tracker = tracker.lock().unwrap();
                        // Update tracker time-based prediction
                        tracker.update(&[], job.dt);
                        tracker.get_predictions()
                    };

                    let latency = start.elapsed().as_secs_f32() * 1000.0;

                    let result = KalmanResult {
                        frame_id: job.frame_id,
                        timestamp: job.timestamp,
                        detections,
                        latency_ms: latency,
                    };

                    if result_tx.send(result).is_err() {
                        break;
                    }
                }

                log::debug!("Kalman worker {} stopped", worker_id);
            });

            workers.push(handle);
        }

        Self {
            job_tx,
            result_rx,
            _workers: workers,
        }
    }

    /// Submit a Kalman prediction job (non-blocking)
    pub fn try_submit(&self, job: KalmanJob) -> bool {
        self.job_tx.try_send(job).is_ok()
    }

    /// Get next result (non-blocking)
    pub fn try_get_result(&self) -> Option<KalmanResult> {
        self.result_rx.try_recv().ok()
    }

    /// Get available capacity (approximate)
    pub fn has_capacity(&self) -> bool {
        self.job_tx.len() < self.job_tx.capacity().unwrap_or(0) / 2
    }
}
