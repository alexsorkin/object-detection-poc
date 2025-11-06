/// Thread-safe detector pool with worker threads submitting to batch executor
/// Workers receive commands and forward them to the batch executor for efficient parallel processing
use crate::batch_executor::{BatchCommand, BatchConfig, BatchExecutor};
use crate::detector_trait::DetectorType;
use crate::types::{Detection, DetectorConfig, ImageData};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::thread;

/// A command to be processed by the detector pool
pub enum DetectorCommand {
    /// Process an image and return detections
    Detect {
        image: ImageData,
        response_tx: Sender<Result<Vec<Detection>, String>>,
    },
    /// Shutdown the worker
    Shutdown,
}

/// Worker pool with multiple worker threads submitting to a single batch executor
pub struct DetectorPool {
    workers: Vec<Worker>,
    executor: Option<BatchExecutor>,
    command_tx: SyncSender<DetectorCommand>,
    batch_tx: SyncSender<BatchCommand>,
    detector_type: DetectorType,
}

impl DetectorPool {
    /// Create a new detector pool with worker threads and batch executor
    pub fn new(
        num_workers: usize,
        detector_type: DetectorType,
        detector_config: DetectorConfig,
        batch_config: BatchConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create BOUNDED channels for backpressure control
        // If channels are full, frames will be dropped (non-blocking behavior)
        let max_queue = batch_config.max_queue_depth;
        let (command_tx, command_rx) = std::sync::mpsc::sync_channel::<DetectorCommand>(max_queue);
        let (batch_tx, batch_rx) = std::sync::mpsc::sync_channel::<BatchCommand>(max_queue);

        // Create batch executor (single thread, processes batches)
        let executor = BatchExecutor::new(batch_rx, detector_type, detector_config, batch_config)?;

        // Create worker threads (forward commands to batch executor)
        let command_rx = std::sync::Arc::new(std::sync::Mutex::new(command_rx));
        let mut workers = Vec::with_capacity(num_workers);

        for id in 0..num_workers {
            let worker = Worker::new(id, std::sync::Arc::clone(&command_rx), batch_tx.clone())?;
            workers.push(worker);
        }

        Ok(DetectorPool {
            workers,
            executor: Some(executor),
            command_tx,
            batch_tx,
            detector_type,
        })
    }

    /// Get the input size expected by the detector (width, height)
    pub fn input_size(&self) -> (u32, u32) {
        match self.detector_type {
            DetectorType::YOLOV8 => (640, 640),
            DetectorType::RTDETR => (640, 640), // RT-DETR now uses 640×640 (same as YOLO)
        }
    }

    /// Submit an image for detection (non-blocking, with backpressure)
    /// Returns None if the queue is full (frame dropped due to backpressure)
    pub fn detect_async(
        &self,
        image: ImageData,
    ) -> Option<Receiver<Result<Vec<Detection>, String>>> {
        let (response_tx, response_rx) = std::sync::mpsc::channel();

        // Use try_send to implement backpressure - drop frame if channel is full
        match self
            .command_tx
            .try_send(DetectorCommand::Detect { image, response_tx })
        {
            Ok(_) => Some(response_rx),
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                // Channel full - drop this frame (backpressure)
                log::debug!("⚠️  Detector queue full, dropping frame (backpressure)");
                None
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                log::error!("❌ Detector pool disconnected");
                None
            }
        }
    }

    /// Submit an image for detection (blocking)
    pub fn detect(&self, image: ImageData) -> Result<Vec<Detection>, String> {
        let response_rx = self
            .detect_async(image)
            .ok_or_else(|| "Detector queue full".to_string())?;
        response_rx
            .recv()
            .map_err(|e| format!("Failed to receive response: {}", e))?
    }
}

impl Drop for DetectorPool {
    fn drop(&mut self) {
        // Send shutdown command to all workers
        for _ in &self.workers {
            let _ = self.command_tx.send(DetectorCommand::Shutdown);
        }

        // Wait for workers to finish
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().ok();
            }
        }

        // Shutdown batch executor
        let _ = self.batch_tx.send(BatchCommand::Shutdown);

        // Wait for executor to finish
        if let Some(executor) = self.executor.take() {
            drop(executor);
        }
    }
}

/// Worker thread that forwards commands to the batch executor
struct Worker {
    #[allow(dead_code)]
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        command_rx: std::sync::Arc<std::sync::Mutex<Receiver<DetectorCommand>>>,
        batch_tx: SyncSender<BatchCommand>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let thread = thread::spawn(move || {
            loop {
                // Lock and receive command
                let command = {
                    let rx = command_rx.lock().unwrap();
                    rx.recv()
                };

                match command {
                    Ok(DetectorCommand::Detect { image, response_tx }) => {
                        // Forward to batch executor (use try_send for backpressure)
                        match batch_tx.try_send(BatchCommand::Detect { image, response_tx }) {
                            Ok(_) => {}
                            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                                log::debug!("⚠️  Batch executor queue full, dropping frame");
                            }
                            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                                log::error!("❌ Batch executor disconnected");
                                break;
                            }
                        }
                    }
                    Ok(DetectorCommand::Shutdown) => {
                        break;
                    }
                    Err(_) => {
                        // Channel closed, exit
                        break;
                    }
                }
            }
        });

        Ok(Worker {
            id,
            thread: Some(thread),
        })
    }
}
