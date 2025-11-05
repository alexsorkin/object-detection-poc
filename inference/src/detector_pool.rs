/// Thread-safe detector pool with worker threads submitting to batch executor
/// Workers receive commands and forward them to the batch executor for efficient parallel processing
use crate::batch_executor::{BatchCommand, BatchConfig, BatchExecutor};
use crate::types::{Detection, DetectorConfig, ImageData};
use std::sync::mpsc::{Receiver, Sender};
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
    command_tx: Sender<DetectorCommand>,
    batch_tx: Sender<BatchCommand>,
}

impl DetectorPool {
    /// Create a new detector pool with worker threads and batch executor
    pub fn new(
        num_workers: usize,
        detector_config: DetectorConfig,
        batch_config: BatchConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create channels
        let (command_tx, command_rx) = std::sync::mpsc::channel::<DetectorCommand>();
        let (batch_tx, batch_rx) = std::sync::mpsc::channel::<BatchCommand>();

        // Create batch executor (single thread, processes batches)
        let executor = BatchExecutor::new(batch_rx, detector_config, batch_config)?;

        // Create worker threads (forward commands to batch executor)
        let command_rx = std::sync::Arc::new(std::sync::Mutex::new(command_rx));
        let mut workers = Vec::with_capacity(num_workers);

        for id in 0..num_workers {
            let worker = Worker::new(
                id,
                std::sync::Arc::clone(&command_rx),
                batch_tx.clone(),
            )?;
            workers.push(worker);
        }

        Ok(DetectorPool {
            workers,
            executor: Some(executor),
            command_tx,
            batch_tx,
        })
    }

    /// Submit an image for detection (non-blocking)
    pub fn detect_async(
        &self,
        image: ImageData,
    ) -> Receiver<Result<Vec<Detection>, String>> {
        let (response_tx, response_rx) = std::sync::mpsc::channel();

        self.command_tx
            .send(DetectorCommand::Detect { image, response_tx })
            .expect("Failed to send command to detector pool");

        response_rx
    }

    /// Submit an image for detection (blocking)
    pub fn detect(&self, image: ImageData) -> Result<Vec<Detection>, String> {
        let response_rx = self.detect_async(image);
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
        batch_tx: Sender<BatchCommand>,
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
                        // Forward to batch executor
                        let _ = batch_tx.send(BatchCommand::Detect { image, response_tx });
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
