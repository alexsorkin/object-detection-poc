/// Frame executor - single-threaded async detection executor
///
/// Executes detection on frames one at a time:
/// - Always processes the latest frame in queue
/// - Drops older frames to stay current
/// - Uses tokio for async execution
use crate::detector_trait::{Detector, DetectorType};
use crate::types::{Detection, DetectorConfig, ImageData};
use std::sync::mpsc::{channel, Receiver, Sender};
use tokio::runtime::Runtime;

/// Configuration for frame execution
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum queue depth before dropping frames (backpressure limit)
    pub max_queue_depth: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        ExecutorConfig {
            max_queue_depth: 2, // Drop frames if more than 2 pending
        }
    }
}

/// A command to be processed by the frame executor
pub enum FrameCommand {
    /// Process an image and return detections
    Detect {
        image: ImageData,
        response_tx: Sender<Result<Vec<Detection>, String>>,
    },
    /// Shutdown the executor
    Shutdown,
}

/// Frame executor that processes detection commands one at a time
pub struct FrameExecutor {
    command_tx: Sender<FrameCommand>,
    #[allow(dead_code)]
    detector_type: DetectorType,
    input_size: (u32, u32),
    #[allow(dead_code)]
    runtime: Runtime,
}

impl FrameExecutor {
    /// Create a new frame executor with tokio runtime
    pub fn new(
        detector_type: DetectorType,
        detector_config: DetectorConfig,
        config: ExecutorConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create channel for commands (unbounded to allow queue inspection)
        let (command_tx, command_rx) = channel::<FrameCommand>();

        // Get input size
        let input_size = match detector_type {
            DetectorType::YOLOV8 => (640, 640),
            DetectorType::RTDETR => (640, 640),
        };

        // Create detector instance
        let detector = detector_type.create(detector_config)?;

        // Create tokio runtime for async execution
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("frame-executor")
            .enable_all()
            .build()?;

        // Spawn executor task
        let max_queue_depth = config.max_queue_depth;
        runtime.spawn(async move {
            Self::executor_loop(detector, command_rx, max_queue_depth).await;
        });

        Ok(FrameExecutor {
            command_tx,
            detector_type,
            input_size,
            runtime,
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
        let (response_tx, response_rx) = channel();

        // Try to send - if channel is disconnected, return None
        match self
            .command_tx
            .send(FrameCommand::Detect { image, response_tx })
        {
            Ok(_) => Some(response_rx),
            Err(_) => {
                log::error!("❌ Frame executor disconnected");
                None
            }
        }
    }

    /// Submit an image for detection (blocking)
    pub fn detect(&self, image: ImageData) -> Result<Vec<Detection>, String> {
        let response_rx = self
            .detect_async(image)
            .ok_or_else(|| "Frame executor disconnected".to_string())?;
        response_rx
            .recv()
            .map_err(|e| format!("Failed to receive response: {}", e))?
    }

    /// Shutdown the executor
    pub fn shutdown(&self) {
        let _ = self.command_tx.send(FrameCommand::Shutdown);
    }

    /// Async executor loop - processes frames one at a time
    async fn executor_loop(
        mut detector: Box<dyn Detector>,
        command_rx: Receiver<FrameCommand>,
        max_queue_depth: usize,
    ) {
        log::info!("Frame executor started");
        let mut frames_processed = 0_u64;
        let mut frames_dropped = 0_u64;

        loop {
            // Collect all pending commands (drop old ones, keep latest)
            let mut commands = Vec::new();

            // Get first command (blocking)
            match command_rx.recv() {
                Ok(cmd) => commands.push(cmd),
                Err(_) => break, // Channel closed
            }

            // Drain any additional pending commands (non-blocking)
            while let Ok(cmd) = command_rx.try_recv() {
                commands.push(cmd);

                // If too many pending, drop older ones
                if commands.len() > max_queue_depth {
                    let dropped = commands.remove(0);
                    if let FrameCommand::Detect { response_tx, .. } = dropped {
                        frames_dropped += 1;
                        let _ = response_tx.send(Err("Frame dropped (too old)".to_string()));
                    }
                }
            }

            // Process only the LATEST command
            if let Some(latest_cmd) = commands.pop() {
                match latest_cmd {
                    FrameCommand::Detect { image, response_tx } => {
                        // Drop all older commands
                        for old_cmd in commands {
                            if let FrameCommand::Detect {
                                response_tx: old_tx,
                                ..
                            } = old_cmd
                            {
                                frames_dropped += 1;
                                let _ = old_tx
                                    .send(Err("Frame dropped (newer frame available)".to_string()));
                            }
                        }

                        // Execute detection on latest frame
                        let result = detector
                            .detect(&image)
                            .map_err(|e| format!("Detection error: {}", e));

                        let _ = response_tx.send(result);
                        frames_processed += 1;

                        if frames_processed % 100 == 0 {
                            log::info!(
                                "Frame executor: {} processed, {} dropped",
                                frames_processed,
                                frames_dropped
                            );
                        }
                    }
                    FrameCommand::Shutdown => {
                        log::info!(
                            "Frame executor shutting down: {} processed, {} dropped",
                            frames_processed,
                            frames_dropped
                        );
                        break;
                    }
                }
            }
        }

        log::info!("Frame executor stopped");
    }
}

impl Drop for FrameExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}
