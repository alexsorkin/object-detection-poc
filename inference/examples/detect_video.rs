/// Real-time Video Detection with Multi-Tracker Support
///
/// Processes video files or camera streams with temporal tracking:
/// - Runs detection pipeline at processing FPS (5-10 FPS based on GPU)
/// - Maintains 30 FPS output using tracker extrapolation
/// - Supports multiple tracking algorithms: Kalman Filter and ByteTrack
/// - Automatically switches to extrapolation when latency > 500ms
///
/// Visual indicators:
/// - GREEN boxes: Real detections from neural network
/// - YELLOW boxes: Tracker extrapolations
///
/// Usage:
///   cargo run --release --features metal --example detect_video [OPTIONS] <video_path>
///
/// Options (can be in any order):
///   --confidence <0-100>    Detection confidence threshold (default: 50)
///   --classes <id,id,...>   Comma-separated class IDs to detect (default: 0,2,3,4,7)
///   --model <variant>       RT-DETR model variant (default: r18_fp32)
///                           Available: r18_fp16, r18_fp32, r34_fp16, r34_fp32, r50_fp16, r50_fp32, r50_int8
///   --tracker <method>      Tracking method: kalman, bytetrack (default: kalman)
///   --headless              Run without display window (default: show window)
///
/// Examples:
///   cargo run --release --features metal --example detect_video test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- --confidence 35 test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- test_data/airport.mp4 --confidence 60 --headless
///   cargo run --release --features metal --example detect_video -- --headless --classes 0,2,5,7 test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- --model r50_fp32 test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- --model r18_fp16 --tracker bytetrack test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- --tracker kalman test_data/airport.mp4
///   cargo run --release --features metal --example detect_video 0  # Use webcam
use image::{Rgb, RgbImage};
use military_target_detector::detector_trait::DetectorType;
use military_target_detector::frame_executor::{ExecutorConfig, FrameExecutor};
use military_target_detector::frame_pipeline::{DetectionPipeline, PipelineConfig};
use military_target_detector::image_utils::{
    calculate_scale_factors, draw_rect_batch, draw_text, draw_text_batch, prepare_annotation_data,
};
use military_target_detector::tracking::{TrackingConfig, TrackingMethod};
use military_target_detector::tracking_types::{ByteTrackConfig, KalmanConfig};
use military_target_detector::types::{DetectorConfig, RTDETRModel};
use military_target_detector::video_pipeline::{Frame, VideoPipeline, VideoPipelineConfig};
use military_target_detector::video_utils::{CaptureInput, VideoResizer};
use opencv::{core::Mat, highgui, imgproc, prelude::*, videoio::VideoWriter};
use std::env;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    // Initialize CUDA globally first (critical for GPU acceleration to work)
    #[cfg(feature = "cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        ort::init()
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build()
            ])
            .commit()
            .map_err(|e| format!("CUDA global init failed: {}", e))?;
    }

    #[cfg(feature = "tensorrt")]
    {
        use ort::execution_providers::TensorRTExecutionProvider;
        ort::init()
            .with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(0)
                    .build()
            ])
            .commit()
            .map_err(|e| format!("TensorRT global init failed: {}", e))?;
    }

    #[cfg(feature = "coreml")]
    {
        use ort::execution_providers::CoreMLExecutionProvider;
        ort::init()
            .with_execution_providers([
                CoreMLExecutionProvider::default()
                    .build()
            ])
            .commit()
            .map_err(|e| format!("CoreML global init failed: {}", e))?;
    }

    #[cfg(feature = "openvino")]
    {
        use ort::execution_providers::OpenVINOExecutionProvider;
        ort::init()
            .with_execution_providers([
                OpenVINOExecutionProvider::default()
                    .build()
            ])
            .commit()
            .map_err(|e| format!("OpenVINO global init failed: {}", e))?;
    }

    #[cfg(feature = "rocm")]
    {
        use ort::execution_providers::ROCmExecutionProvider;
        ort::init()
            .with_execution_providers([
                ROCmExecutionProvider::default()
                    .with_device_id(0)
                    .build()
            ])
            .commit()
            .map_err(|e| format!("ROCm global init failed: {}", e))?;
    }

    eprintln!("📝 Parsing arguments...");

    let args: Vec<String> = env::args().collect();
    eprintln!("   Args: {:?}", args);
    io::stderr().flush()?;

    // Parse confidence threshold, class filter, and display mode
    let mut confidence_threshold = 0.50; // Default 50%
    let mut allowed_classes: Vec<u32> = vec![0, 2, 3, 4, 7]; // person, car, motorcycle, airplane, truck
    let mut headless = false; // Default: show display window
    let mut model_variant = RTDETRModel::R50VD_FP32; // Default: r18_fp32
    let mut tracking_method = TrackingMethod::ByteTrack; // Default: Kalman
    let mut detector_type = DetectorType::RTDETR; // Default: RT-DETR
    let mut video_source: Option<String> = None;

    // Parse all arguments in any order
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--confidence" => {
                i += 1;
                if i < args.len() {
                    match args[i].parse::<f32>() {
                        Ok(val) => {
                            confidence_threshold = (val / 100.0).clamp(0.0, 1.0);
                        }
                        Err(_) => {
                            eprintln!("Invalid confidence threshold. Use a number between 0-100");
                            return Ok(());
                        }
                    }
                }
            }
            "--classes" => {
                i += 1;
                if i < args.len() {
                    allowed_classes = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse::<u32>().ok())
                        .collect();
                    if allowed_classes.is_empty() {
                        eprintln!("Invalid classes. Use comma-separated numbers (e.g., 0,2,5,7)");
                        return Ok(());
                    }
                }
            }
            "--headless" => {
                headless = true;
            }
            "--model" => {
                i += 1;
                if i < args.len() {
                    match RTDETRModel::from_str(&args[i]) {
                        Some(model) => {
                            model_variant = model;
                        }
                        None => {
                            eprintln!(
                                "Invalid model variant '{}'. Valid options: r18_fp16, r18_fp32, r34_fp16, r34_fp32, r50_fp16, r50_fp32, r50_uint8",
                                args[i]
                            );
                            return Ok(());
                        }
                    }
                }
            }
            "--tracker" => {
                i += 1;
                if i < args.len() {
                    match args[i].to_lowercase().as_str() {
                        "kalman" => {
                            tracking_method = TrackingMethod::Kalman;
                        }
                        "bytetrack" => {
                            tracking_method = TrackingMethod::ByteTrack;
                        }
                        _ => {
                            eprintln!(
                                "Invalid tracking method '{}'. Valid options: kalman, bytetrack",
                                args[i]
                            );
                            return Ok(());
                        }
                    }
                }
            }
            "--detector" => {
                i += 1;
                if i < args.len() {
                    match args[i].to_lowercase().as_str() {
                        "rtdetr" | "rt-detr" => {
                            detector_type = DetectorType::RTDETR;
                        }
                        _ => {
                            log::warn!(
                                "⚠️  Unknown detector type: '{}'. Supported: [rtdetr]",
                                args[i]
                            );
                        }
                    }
                }
            }
            arg => {
                // If it doesn't start with --, treat it as the video source
                if !arg.starts_with("--") {
                    video_source = Some(arg.to_string());
                }
            }
        }
        i += 1;
    }

    let video_source = video_source.unwrap_or_else(|| {
        eprintln!("ℹ️  No video path provided, using default: test_data/airport.mp4");
        "test_data/airport.mp4".to_string()
    });

    eprint!("📹 Opening video source: {}... ", video_source);
    io::stderr().flush()?;
    let last_fps = Arc::new(Mutex::new(20.0_f32));
    let fps = Arc::clone(&last_fps); // Default FPS, will be updated by actual video

    // Read model directory from environment variable or use default
    let model_dir = env::var("DEFENITY_MODEL_DIR").unwrap_or_else(|_| "../models".to_string());

    eprintln!("🤖 Using detector: {:?}", detector_type);

    let detector_config = DetectorConfig {
        model_path: format!("{}/{}", model_dir, model_variant.filename()),
        confidence_threshold,
        input_size: (640, 640),
        use_gpu: true, // Use CPU or GPU (CUDA/Metal/OpenVINO/Vulkan)
        ..Default::default()
    };

    let executor_config = ExecutorConfig {
        max_queue_depth: 1, // Drop frames if more than 1 pending (backpressure control)
    };

    let frame_executor = Arc::new(FrameExecutor::new(
        detector_type,
        detector_config.clone(),
        executor_config.clone(),
    )?);

    // Create detection pipeline
    let pipeline_config = PipelineConfig {
        tile_overlap: 32,
        allowed_classes: allowed_classes.clone(),
        iou_threshold: 0.5,
    };

    let pipeline = Arc::new(DetectionPipeline::new(
        Arc::clone(&frame_executor),
        pipeline_config,
    ));

    // Create video pipeline with selected tracker
    let buffer_size = 1; // Tiny buffer - only allow 2 frames queued max

    // Configure tracking based on selected method
    let tracking_config = match tracking_method {
        TrackingMethod::Kalman => {
            TrackingConfig::Kalman(KalmanConfig {
                max_age: 25,        // frames to keep track alive without detections (30*20ms = 600ms)
                min_hits: 3,        // REDUCED from 3 to test track position updates
                iou_threshold: 0.3, // REDUCED from 0.3 to be more permissive for association
                init_tracker_min_score: 0.25, // minimum confidence to create new track (25% - standard value)
                measurement_noise: [1.0, 1.0, 10.0, 10.0], // measurement noise covariance
                process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001], // standard process noise
                maintenance_period_ms: 50,    // maintenance period in milliseconds
            })
        }
        TrackingMethod::ByteTrack => {
            TrackingConfig::ByteTrack(ByteTrackConfig {
                max_age: 25,                               // frames to keep track alive (30*20ms = 600ms)
                min_hits: 3,        // REDUCED from 3 to test track position updates
                iou_threshold: 0.3, // IoU threshold for association (standard value)
                init_tracker_min_score: 0.25, // minimum confidence to create new track (25% - standard value)
                measurement_noise: [1.0, 1.0, 10.0, 10.0], // measurement noise
                process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001], // process noise
                high_score_threshold: 0.5,    // high confidence threshold
                low_score_threshold: 0.1,     // low confidence threshold
                maintenance_period_ms: 50,    // maintenance period in milliseconds
            })
        }
    };

    let video_config = VideoPipelineConfig {
        tracking_config,
        buffer_size,
    };

    let video_pipeline = Arc::new(VideoPipeline::new(Arc::clone(&pipeline), video_config));

    // Create shutdown channels for coordinating thread shutdown
    let (shutdown_tx_capture, shutdown_rx_capture) = sync_channel::<()>(1);
    let (shutdown_tx_detector, shutdown_rx_detector) = sync_channel::<()>(1);
    let (shutdown_tx_stats, shutdown_rx_stats) = sync_channel::<()>(1);
    let (shutdown_tx_writer, shutdown_rx_writer) = sync_channel::<()>(1);
    let (shutdown_tx_output, shutdown_rx_output) = sync_channel::<()>(1);
    let (shutdown_tx_display, shutdown_rx_display) = sync_channel::<()>(1);

    eprintln!("  ✓ Pipeline ready");
    eprintln!("\n💡 Configuration:");
    eprintln!("  • Detector: {} (frame executor)", model_variant.name());
    eprintln!(
        "  • Queue depth: {} (backpressure)",
        executor_config.max_queue_depth
    );
    eprintln!("  • Confidence: {:.0}%", confidence_threshold * 100.0);
    eprintln!("  • Classes: {:?}", allowed_classes);
    eprintln!("  • Tracker: {}", tracking_method);
    eprintln!(
        "  • Display mode: {}",
        if headless {
            "headless (no window)"
        } else {
            "window enabled"
        }
    );

    eprintln!("\n🎬 Starting video processing...");
    if !headless {
        eprintln!("  Press 'q' to quit, 'p' to pause\n");
    } else {
        eprintln!("  Press Ctrl+C to stop\n");
    }
    io::stderr().flush()?;

    // We'll create the video writer after we get the first frame (to know actual dimensions)
    let output_path = "output_video.mp4";

    // Create window for display - will be resized after first frame to match video size (max 640p)
    if !headless {
        highgui::named_window("Detection", highgui::WINDOW_NORMAL)?;
    }

    let mut paused = false;

    // Create clones of Arc for threads - these will all share the same atomic
    let stats_processed = Arc::new(AtomicU32::new(0));
    let stats_extrapolated = Arc::new(AtomicU32::new(0));
    let stats_latency_sum = Arc::new(AtomicU32::new(0)); // Store as integer (sum of ms)
    let stats_start = Instant::now();

    // Create async capture thread - captures frames at video's native FPS
    // TWO separate queues:
    // 1. detector_rx: Can drop frames (non-blocking), used for detection - carries timestamp
    // 2. output_rx: Must process ALL frames (blocking), used for output
    let (detector_tx, detector_rx) = sync_channel::<(Arc<RgbImage>, Instant)>(5); // Detector queue with Arc to avoid cloning
    let (output_tx, output_rx) = sync_channel::<(Arc<RgbImage>, Instant)>(5); // Output queue with Arc to avoid cloning
    let (writer_tx, writer_rx) = sync_channel::<(Mat, u32, u32)>(100);
    let (display_tx, display_rx) = sync_channel::<(Mat, u32, u32)>(100);

    // Create channels for VideoResizer output
    let (capture_frame_tx, capture_frame_rx) = crossbeam::channel::unbounded::<CaptureInput>();

    // Start VideoResizer thread
    let video_source_clone = video_source.clone();
    let capture_handle = thread::spawn(move || {
        let resizer = VideoResizer::new();

        let result = if let Ok(cam_id) = video_source_clone.parse::<i32>() {
            log::info!("Starting camera capture from device {}", cam_id);
            resizer.resize_camera(cam_id, capture_frame_tx)
        } else {
            log::info!("Starting file/stream capture from {}", video_source_clone);
            resizer.resize_stream(&video_source_clone, capture_frame_tx)
        };

        if let Err(e) = result {
            log::error!("Video capture error: {}", e);
        }

        log::debug!("Capture thread: Ending");
    });

    // Create frame conversion thread - converts Mat to RgbImage and sends to detector/output queues
    let shutdown_rx_converter = shutdown_rx_capture;
    let fps_for_converter = Arc::clone(&fps);
    let converter_handle = thread::spawn(move || {
        let mut cap_frame_id = 0_u64;

        loop {
            // Check for shutdown signal first
            if shutdown_rx_converter.try_recv().is_ok() {
                log::debug!("Converter thread received shutdown signal");
                break;
            }

            // Blocking receive - wait for next frame
            match capture_frame_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(capture) => {
                    let capture_time = Instant::now();
                    *fps_for_converter.lock().unwrap() = capture.fps as f32;

                    log::debug!(
                        "Dispatching frame {}, FPS: {:.2}",
                        cap_frame_id,
                        capture.fps
                    );

                    // OPTIMIZATION: Images are already Arc-wrapped from CaptureInput, use cheap Arc::clone
                    output_tx
                        .try_send((Arc::clone(&capture.original_image), capture_time))
                        .ok();
                    // Send to detector queue (NON-BLOCKING)
                    detector_tx
                        .try_send((Arc::clone(&capture.resized_image), capture_time))
                        .ok();

                    if !capture.has_frames {
                        log::info!("No frames has left in stream, exiting..");
                        shutdown_tx_display.send(()).ok();
                    }

                    cap_frame_id += 1;
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    // No frame available, continue loop
                    continue;
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    log::info!("Capture channel disconnected, converter thread exiting");
                    break;
                }
            }
        }

        log::debug!("Converter thread: Closing channels");
    });

    let video_pipeline_for_detector = Arc::clone(&video_pipeline);
    let fps_for_detector = Arc::clone(&fps);

    let detector_handle = thread::spawn(move || {
        let mut submitted_frame_id = 0_u64;
        let mut input_frame_id = 0_u64;

        loop {
            let mut latest_detector_frame: Option<Arc<RgbImage>> = None;
            let mut frames_drained = 0_i32;
            let frame_start = Instant::now();

            // Drain all pending frames, keep only the latest
            while let Ok((detector_frame, _capture_time)) = detector_rx.try_recv() {
                // If we already have a frame, the previous one becomes a dropped frame
                if latest_detector_frame.is_some() {
                    frames_drained += 1;
                }
                latest_detector_frame = Some(detector_frame);
                input_frame_id += 1;
            }

            if frames_drained > 0 {
                log::warn!("Drained frames: {} (behind schedule)", frames_drained);
            }

            if latest_detector_frame.is_some() {
                log::debug!("Consuming frame {} for detect", input_frame_id);
                // OPTIMIZATION: detector_frame is Arc<RgbImage>, dereference to access methods
                let detector_frame = latest_detector_frame.unwrap();
                let (width, height) = detector_frame.dimensions();

                // OPTIMIZATION: Only clone pixel data when converting to owned Vec for Frame
                // This is the minimal necessary allocation for the detection pipeline
                let raw_data: Vec<u8> = match Arc::try_unwrap(detector_frame) {
                    Ok(img) => img.into_raw(),
                    Err(arc) => arc.as_ref().to_vec(),
                };

                let frame = Frame {
                    data: Arc::from(raw_data.into_boxed_slice()), // Wrap in Arc for zero-copy sharing
                    width,
                    height,
                    sequence: submitted_frame_id,
                };

                // submit_frame is NON-BLOCKING - if pipeline is busy, frame is dropped
                // This provides natural backpressure without pacing
                match video_pipeline_for_detector.submit_frame(frame) {
                    Ok(()) => {
                        log::debug!("Submitted frame {} for detect", submitted_frame_id);
                        submitted_frame_id += 1;
                    }
                    Err(_) => {
                        log::warn!("Dropped frame {} (pipeline is busy)", input_frame_id);
                        continue;
                    }
                };
            }

            // Pace to target FPS - sleep for remaining time in this frame period
            // This happens AFTER draining and submitting to avoid queue buildup
            let capture_frame_duration =
                Duration::from_secs_f64(1.0 / *fps_for_detector.lock().unwrap() as f64);
            let elapsed = frame_start.elapsed();
            if elapsed < capture_frame_duration {
                std::thread::sleep(capture_frame_duration - elapsed);
            }

            // Check for shutdown signal (non-blocking)
            if shutdown_rx_detector.try_recv().is_ok() {
                log::debug!("Detector thread received shutdown signal");
                video_pipeline_for_detector.shutdown();
                break;
            }
        }
    });

    // Create stats collector thread - consumes tracker predictions asynchronously
    // This prevents stats retrieval from blocking the main display loop
    let video_pipeline_for_stats = Arc::clone(&video_pipeline);
    let stats_processed_for_stats = Arc::clone(&stats_processed);
    let stats_extrapolated_for_stats = Arc::clone(&stats_extrapolated);
    let stats_latency_for_stats = Arc::clone(&stats_latency_sum);

    let stats_handle = thread::spawn(move || {
        loop {
            // Non-blocking check for stats
            match video_pipeline_for_stats.get_result() {
                Some(stats) => {
                    // Update stats
                    if stats.is_extrapolated {
                        let new_extrapolated =
                            stats_extrapolated_for_stats.fetch_add(1, Ordering::Relaxed) + 1;
                        log::debug!(
                            "Got extrapolated stats, total extrapolated: {}",
                            new_extrapolated
                        );
                    } else {
                        let new_processed =
                            stats_processed_for_stats.fetch_add(1, Ordering::Relaxed) + 1;
                        log::debug!("Got processed stats, total processed: {}", new_processed);
                    }
                    stats_latency_for_stats.fetch_add(stats.latency_ms as u32, Ordering::Relaxed);
                }
                None => {
                    // No stats available yet
                }
            }

            // Check for shutdown signal (non-blocking)
            if shutdown_rx_stats.try_recv().is_ok() {
                log::debug!("Results thread received shutdown signal");
                break;
            }

            // Small sleep to avoid busy loop
            std::thread::sleep(Duration::from_millis(5));
        }
    });

    let fps_for_writer = Arc::clone(&fps);
    let writer_handle = thread::spawn(move || {
        let mut file_writer: Option<VideoWriter> = None;

        loop {
            while let Ok((mat, width, height)) = writer_rx.try_recv() {
                if let Some(writer) = &mut file_writer {
                    log::trace!("Writing frame to output video file");
                    let _ = writer.write(&mat);
                } else {
                    file_writer = match VideoWriter::new(
                        output_path,
                        VideoWriter::fourcc('a', 'v', 'c', '1').unwrap(),
                        *fps_for_writer.lock().unwrap() as f64,
                        opencv::core::Size::new(width as i32, height as i32),
                        true,
                    ) {
                        Ok(writer) => Some(writer),
                        Err(e) => {
                            eprintln!("❌ Failed to open video writer: {}", e);
                            break;
                        }
                    };
                    log::debug!("Writing first frame to output video file");
                    let _ = file_writer.as_mut().unwrap().write(&mat);
                }
            }
            if shutdown_rx_writer.try_recv().is_ok() {
                log::debug!("Writer thread received shutdown signal");
                break;
            }
        }
        if let Some(writer) = &mut file_writer {
            log::debug!("Releasing video writer");
            let _ = writer.release().map_err(|e| e.to_string());
            log::info!("Output video saved: {}", output_path);
        }
    });

    let fps_for_output = Arc::clone(&fps);
    let output_handle = thread::spawn(move || {
        let mut out_frame_id = 0_u64;
        let mut elapsed = stats_start.elapsed().as_secs_f32();
        let mut avg_fps = 10.0 as f32 / elapsed;
        let mut processed = stats_processed.load(Ordering::Relaxed);
        let mut extrapolated = stats_extrapolated.load(Ordering::Relaxed);
        let mut total = processed + extrapolated;
        let mut avg_latency = if total > 0 {
            stats_latency_sum.load(Ordering::Relaxed) as f32 / total as f32
        } else {
            0.0
        };

        loop {
            // Try to get frame from OUTPUT queue with timeout
            // This allows the loop to check for exit conditions periodically
            // instead of blocking forever waiting for frames
            let frame_arc = match output_rx.recv_timeout(Duration::from_millis(20)) {
                Ok((cap_arc, _timestamp)) => {
                    log::debug!("Consuming frame: {} for output", out_frame_id);
                    out_frame_id += 1;
                    cap_arc
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No frame yet, continue loop to check exit conditions
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed - capture thread finished
                    log::debug!("Output channel closed, ending video processing");
                    break;
                }
            };

            let origin_frame_width = frame_arc.width();
            let origin_frame_height = frame_arc.height();

            // === OPTIMIZATION: Pre-compute scale factors ONCE (after first frame) ===
            // These don't change between frames, so calculate once and reuse
            let (scale_x, scale_y) =
                calculate_scale_factors(origin_frame_width, origin_frame_height, 640.0);

            // Validate predictions before annotation to catch issues early
            let valid_predictions: Vec<_> = video_pipeline
                .get_predictions()
                .iter()
                .filter(|d| {
                    // Check for valid bounding box dimensions
                    if d.w <= 0.0 || d.h <= 0.0 {
                        log::warn!(
                            "Invalid prediction bbox - width: {}, height: {}, skipping",
                            d.w,
                            d.h
                        );
                        return false;
                    }

                    // Check for very small dimensions that might cause issues
                    if d.w < 1.0 || d.h < 1.0 {
                        log::warn!(
                            "Too small prediction bbox - width: {}, height: {}, skipping",
                            d.w,
                            d.h
                        );
                        return false;
                    }

                    true
                })
                .cloned()
                .collect();

            let num_tracks = valid_predictions
                .iter()
                .filter(|d| d.track_id.is_some())
                .count();

            // Annotate current frame with latest tracker predictions (from VideoPipeline with tracking)
            let mut annotated = frame_arc.as_ref().clone(); // Clone only once when we need to annotate

            // PARALLEL: Pre-compute all annotation data (boxes, labels, colors)
            let annotation_data: Vec<_> =
                prepare_annotation_data(&valid_predictions, scale_x, scale_y);

            // PARALLEL BATCH: Prepare rectangle and label data for batch drawing
            let rects: Vec<_> = annotation_data
                .iter()
                .map(|(x, y, w, h, color, _label, _show_label)| (*x, *y, *w, *h, *color, 2))
                .collect();

            let labels: Vec<_> = annotation_data
                .iter()
                .filter(|(_x, _y, _w, _h, _color, _label, show_label)| *show_label)
                .map(|(x, _y, _w, _h, _color, label, _show_label)| {
                    (label.as_str(), x + 3, _y + 15, Rgb([255, 255, 255]), None)
                })
                .collect();

            // Draw all rectangles and labels in parallel
            draw_rect_batch(&mut annotated, &rects);
            draw_text_batch(&mut annotated, &labels);

            // Draw stats overlay
            {
                let stats_text = format!(
                    "Frame: {} # Tracks: {} # RT-DETR Runs: {}",
                    out_frame_id,
                    num_tracks,
                    stats_processed.load(Ordering::Relaxed)
                );

                draw_text(
                    &mut annotated,
                    &stats_text,
                    10,
                    30,
                    Rgb([255, 255, 255]),
                    Some(Rgb([0, 0, 0])),
                );
            }

            // Convert annotated frame to Mat for display and writing
            let display_width = annotated.width();
            let display_height = annotated.height();

            // Create Mat from RGB data - opencv expects CV_8UC3 format
            let data_slice = annotated.as_raw();
            let mat = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    display_height as i32,
                    display_width as i32,
                    opencv::core::CV_8UC3,
                    data_slice.as_ptr() as *mut _,
                    opencv::core::Mat_AUTO_STEP,
                )
            }
            .unwrap();

            let mut display_mat = Mat::default();

            // Reuse pre-allocated display_mat buffer for BGR conversion
            let _ = imgproc::cvt_color(
                &mat,
                &mut display_mat,
                imgproc::COLOR_RGB2BGR,
                0,
            );

            let _ =
                display_tx.try_send((display_mat.clone(), frame_arc.width(), frame_arc.height()));
            // Write ALL frames to output video (async, non-blocking)
            let _ =
                writer_tx.try_send((display_mat.clone(), frame_arc.width(), frame_arc.height()));

            // Print progress every N frames (based on FPS)
            let progress_interval = (*fps_for_output.lock().unwrap() as u64).max(24);
            if out_frame_id % progress_interval == 0 {
                elapsed = stats_start.elapsed().as_secs_f32();
                avg_fps = out_frame_id as f32 / elapsed;
                processed = stats_processed.load(Ordering::Relaxed);
                extrapolated = stats_extrapolated.load(Ordering::Relaxed);
                total = processed + extrapolated;
                avg_latency = if total > 0 {
                    stats_latency_sum.load(Ordering::Relaxed) as f32 / total as f32
                } else {
                    0.0
                };

                log::debug!(
                    "Checkpoint: {:.1} FPS | Tracks: {} | Display frames: {}",
                    avg_fps,
                    num_tracks,
                    out_frame_id,
                );

                if total == 0 {
                    log::warn!(
                    "⚠️  No detection stats yet (processed: {}, extrapolated: {}) - check for ONNX errors",
                    processed, extrapolated
                );
                } else {
                    log::debug!(
                        "RT-DETR runs: {} | Extrapolated: {} | Avg Latency: {:.0}ms",
                        processed,
                        extrapolated,
                        avg_latency
                    );
                }
                let _ = io::stderr().flush();
            }

            // Check shutdown signal
            if shutdown_rx_output.try_recv().is_ok() {
                log::debug!("Output thread received shutdown signal");
                break;
            }
        }
        // Shutdown video pipeline
        video_pipeline.shutdown();

        eprintln!("\n");
        eprintln!(
            "==============================================================================="
        );
        eprintln!(
            "Finished with: frame_id={}, total_time={}, processed={}, extrapolated={}",
            out_frame_id, elapsed, processed, extrapolated
        );
        eprintln!(
            "==============================================================================="
        );

        eprintln!("\n");
        eprintln!("===================================================");
        eprintln!("FINAL STATISTICS:");
        eprintln!("===================================================");
        eprintln!("Total frames displayed: {}", out_frame_id);
        eprintln!("Duration: {:.1}s", elapsed);
        eprintln!("Average FPS: {:.1}", avg_fps);
        eprintln!("");
        eprintln!("Detection Stats:");
        eprintln!("  RT-DETR runs: {}", processed);
        eprintln!("  Average latency: {:.0}ms", avg_latency);
        eprintln!("===================================================\n");
        let _ = io::stderr().flush();
    });

    // Pre-allocate Mat for reuse (optimization)
    let display_initialized = std::cell::Cell::new(false);
    let mut latest_mat = Mat::default();

    loop {
        log::debug!("Main Loop: Starting iteration, checking for frames...");

        if !headless {
            latest_mat = match display_rx.try_recv() {
                Ok((mat, width, height)) => {
                    if !display_initialized.get() {
                        let max_height = 640;
                        let (window_width, window_height) = if height > max_height {
                            // Scale down to 640p preserving aspect ratio
                            let scale = max_height as f32 / height as f32;
                            ((width as f32 * scale) as i32, max_height as i32)
                        } else {
                            // Use native resolution
                            (width as i32, height as i32)
                        };
                        let _ = highgui::resize_window("Detection", window_width, window_height);
                        display_initialized.set(true);
                    }
                    // drain one more frame if available to reduce latency
                    match display_rx.try_recv() {
                        Ok((next_mat, _, _)) => next_mat,
                        Err(_) => mat,
                    }
                }
                // If no new frame, reuse last one
                Err(_) => latest_mat,
            };

            let _ = highgui::imshow("Detection", &latest_mat);

            // Handle keyboard input with minimal delay (1ms for UI responsiveness)
            let key = highgui::wait_key(1)?;
            if key == 'q' as i32 {
                eprintln!("\n⏹️  Stopped by user");
                io::stderr().flush()?;
                break;
            } else if key == 'p' as i32 {
                paused = !paused;
                eprintln!("\n⏸️  Paused: {}", paused);
                io::stderr().flush()?;
            }

            if paused {
                while highgui::wait_key(100)? != 'p' as i32 {}
                paused = false;
                eprintln!("▶️  Resumed");
                io::stderr().flush()?;
            }
        }

        if shutdown_rx_display.try_recv().is_ok() {
            log::debug!("Main loop received shutdown signal");
            if !headless {
                let _ = highgui::destroy_all_windows();
            }
            break;
        }
    }

    // Send shutdown signals to all threads when main loop ends
    log::info!("Main loop ended, shutting down all threads");
    let _ = shutdown_tx_capture.send(()).is_ok() && converter_handle.join().is_ok();
    let _ = capture_handle.join().is_ok();
    let _ = shutdown_tx_detector.send(()).is_ok() && detector_handle.join().is_ok();
    let _ = shutdown_tx_stats.send(()).is_ok() && stats_handle.join().is_ok();
    let _ = shutdown_tx_writer.send(()).is_ok() && writer_handle.join().is_ok();
    let _ = shutdown_tx_output.send(()).is_ok() && output_handle.join().is_ok();

    Ok(())
}
