/// Batch executor that collects detection commands and executes them in batches
/// based on configurable batch size and timeout thresholds.
use crate::detector_trait::{Detector, DetectorType};
use crate::types::{Detection, DetectorConfig, ImageData};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for batch execution
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of images to batch together
    pub batch_size: usize,
    /// Maximum time to wait before executing a partial batch (in milliseconds)
    pub timeout_ms: u64,
    /// Maximum queue depth before dropping frames (backpressure limit)
    pub max_queue_depth: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            batch_size: 4,      // Process up to 4 tiles at once
            timeout_ms: 50,     // Wait max 50ms for batch to fill
            max_queue_depth: 2, // Drop frames if more than 2 pending
        }
    }
}

/// A command to be processed by the batch executor
pub enum BatchCommand {
    /// Process an image and return detections
    Detect {
        image: ImageData,
        response_tx: Sender<Result<Vec<Detection>, String>>,
    },
    /// Shutdown the executor
    Shutdown,
}

/// Batch item containing the command and metadata
struct BatchItem {
    image: ImageData,
    response_tx: Sender<Result<Vec<Detection>, String>>,
}

/// Batch executor that processes detection commands in batches
pub struct BatchExecutor {
    command_tx: SyncSender<BatchCommand>,
    thread: Option<thread::JoinHandle<()>>,
    #[allow(dead_code)]
    detector_type: DetectorType,
    input_size: (u32, u32),
}

impl BatchExecutor {
    /// Create a new batch executor with internal channel for backpressure control
    pub fn new(
        detector_type: DetectorType,
        detector_config: DetectorConfig,
        batch_config: BatchConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create bounded channel for backpressure
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<BatchCommand>(batch_config.max_queue_depth);

        // Get input size before moving detector_type
        let input_size = match detector_type {
            DetectorType::YOLOV8 => (640, 640),
            DetectorType::RTDETR => (640, 640),
        };

        // Create detector instance based on type
        let mut detector = detector_type.create(detector_config)?;

        let thread = thread::spawn(move || {
            let mut batch: Vec<BatchItem> = Vec::with_capacity(batch_config.batch_size);
            let mut batch_start: Option<Instant> = None;

            loop {
                // Calculate timeout for receiving next command
                let timeout = if batch.is_empty() {
                    // No pending batch, wait indefinitely
                    None
                } else {
                    // Pending batch, calculate remaining time
                    let elapsed = batch_start.unwrap().elapsed();
                    let timeout_duration = Duration::from_millis(batch_config.timeout_ms);
                    if elapsed >= timeout_duration {
                        // Timeout already expired, process immediately
                        Some(Duration::from_millis(0))
                    } else {
                        Some(timeout_duration - elapsed)
                    }
                };

                // Try to receive next command with timeout
                let command_result = if let Some(timeout_duration) = timeout {
                    command_rx.recv_timeout(timeout_duration)
                } else {
                    command_rx
                        .recv()
                        .map_err(|_| std::sync::mpsc::RecvTimeoutError::Disconnected)
                };

                match command_result {
                    Ok(BatchCommand::Detect { image, response_tx }) => {
                        // Add to batch
                        if batch.is_empty() {
                            batch_start = Some(Instant::now());
                        }
                        batch.push(BatchItem { image, response_tx });

                        // Execute if batch is full
                        if batch.len() >= batch_config.batch_size {
                            Self::execute_batch(&mut detector, &mut batch);
                            batch_start = None;
                        }
                    }
                    Ok(BatchCommand::Shutdown) => {
                        // Execute any pending batch before shutting down
                        if !batch.is_empty() {
                            Self::execute_batch(&mut detector, &mut batch);
                        }
                        break;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // Timeout expired, execute pending batch
                        if !batch.is_empty() {
                            Self::execute_batch(&mut detector, &mut batch);
                            batch_start = None;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        // Channel closed, execute pending batch and exit
                        if !batch.is_empty() {
                            Self::execute_batch(&mut detector, &mut batch);
                        }
                        break;
                    }
                }
            }
        });

        Ok(BatchExecutor {
            command_tx,
            thread: Some(thread),
            detector_type,
            input_size,
        })
    }

    /// Get the input size expected by the detector (width, height)
    pub fn input_size(&self) -> (u32, u32) {
        self.input_size
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
            .try_send(BatchCommand::Detect { image, response_tx })
        {
            Ok(_) => Some(response_rx),
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                log::debug!("⚠️  Batch executor queue full, dropping frame (backpressure)");
                None
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                log::error!("❌ Batch executor disconnected");
                None
            }
        }
    }

    /// Submit an image for detection (blocking)
    pub fn detect(&self, image: ImageData) -> Result<Vec<Detection>, String> {
        let response_rx = self
            .detect_async(image)
            .ok_or_else(|| "Batch executor queue full".to_string())?;
        response_rx
            .recv()
            .map_err(|e| format!("Failed to receive response: {}", e))?
    }

    /// Execute a batch of detections using true batch inference (if batch_size=1) or sequential
    fn execute_batch(detector: &mut Box<dyn Detector>, batch: &mut Vec<BatchItem>) {
        if batch.is_empty() {
            return;
        }

        let batch_start = Instant::now();
        let batch_size = batch.len();

        // Check if we can do true batch inference
        if batch.len() == 1 {
            // Single image - use regular detect
            let item = batch.pop().unwrap();
            let result = detector
                .detect(&item.image)
                .map_err(|e| format!("Detection error: {}", e));
            let _ = item.response_tx.send(result);
            let duration = batch_start.elapsed();
            log::info!(
                "🔥 GPU Batch Execution: {} image in {:.1}ms ({:.1} FPS)",
                batch_size,
                duration.as_secs_f32() * 1000.0,
                1000.0 / (duration.as_secs_f32() * 1000.0)
            );
            return;
        }

        // Try batch inference for multiple images
        let images: Vec<_> = batch.iter().map(|item| item.image.clone()).collect();

        let results = detector.detect_batch(&images);

        match results {
            Ok(all_detections) => {
                let duration = batch_start.elapsed();
                log::info!(
                    "🔥 GPU Batch Execution: {} images in {:.1}ms ({:.1} FPS per image)",
                    batch_size,
                    duration.as_secs_f32() * 1000.0,
                    1000.0 / (duration.as_secs_f32() * 1000.0 / batch_size as f32)
                );

                // Send individual results back to each requester
                for (item, detections) in batch.drain(..).zip(all_detections.into_iter()) {
                    let _ = item.response_tx.send(Ok(detections));
                }
            }
            Err(e) => {
                // If batch inference fails (model doesn't support dynamic batch size),
                // fall back to sequential processing
                if e.to_string().contains("Got invalid dimensions")
                    || e.to_string().contains("Expected: 1")
                {
                    log::warn!(
                        "⚠️  Batch inference not supported, falling back to sequential processing"
                    );
                    let sequential_start = Instant::now();
                    // Model requires batch_size=1, process sequentially
                    for item in batch.drain(..) {
                        let result = detector
                            .detect(&item.image)
                            .map_err(|e| format!("Detection error: {}", e));
                        let _ = item.response_tx.send(result);
                    }
                    let duration = sequential_start.elapsed();
                    log::info!(
                        "🔥 Sequential Execution: {} images in {:.1}ms ({:.1} FPS per image)",
                        batch_size,
                        duration.as_secs_f32() * 1000.0,
                        1000.0 / (duration.as_secs_f32() * 1000.0 / batch_size as f32)
                    );
                } else {
                    // Other error, send to all requesters
                    let error_msg = format!("Batch detection error: {}", e);
                    for item in batch.drain(..) {
                        let _ = item.response_tx.send(Err(error_msg.clone()));
                    }
                }
            }
        }
    }
}

impl Drop for BatchExecutor {
    fn drop(&mut self) {
        // Send shutdown command
        let _ = self.command_tx.send(BatchCommand::Shutdown);

        // Wait for executor thread to finish
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}
