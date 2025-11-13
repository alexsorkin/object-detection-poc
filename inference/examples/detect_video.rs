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
///                           Available: r18_fp16, r18_fp32, r34_fp16, r34_fp32, r50_fp16, r50_fp32
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
use opencv::{
    core::Mat,
    highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use std::env;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Helper function to capture and convert a frame from video
fn capture_frame(cap: &mut VideoCapture) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let mut mat_frame = Mat::default();
    if !cap.read(&mut mat_frame)? || mat_frame.empty() {
        return Err("End of video".into());
    }

    // Convert BGR to RGB
    let mut rgb_mat = Mat::default();
    imgproc::cvt_color(
        &mat_frame,
        &mut rgb_mat,
        imgproc::COLOR_BGR2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Convert to RgbImage
    let width = rgb_mat.cols() as u32;
    let height = rgb_mat.rows() as u32;
    let data = rgb_mat.data_bytes()?.to_vec();
    let rgb_image = RgbImage::from_vec(width, height, data).ok_or("Failed to create RgbImage")?;

    Ok(rgb_image)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};

    // Print immediately before anything else
    eprintln!("🚀 Starting detect_video example...");
    io::stderr().flush()?;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    eprintln!("📝 Parsing arguments...");
    io::stderr().flush()?;

    let args: Vec<String> = env::args().collect();
    eprintln!("   Args: {:?}", args);
    io::stderr().flush()?;

    // Parse confidence threshold, class filter, and display mode
    let mut confidence_threshold = 0.50; // Default 50%
    let mut allowed_classes: Vec<u32> = vec![0, 2, 3, 4, 7]; // person, car, motorcycle, airplane, truck
    let mut headless = false; // Default: show display window
    let mut model_variant = RTDETRModel::R18VD_FP32; // Default: r18_fp32
    let mut tracking_method = TrackingMethod::Kalman; // Default: Kalman
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
                                "Invalid model variant '{}'. Valid options: r18_fp16, r18_fp32, r34_fp16, r34_fp32, r50_fp16, r50_fp32",
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

    eprintln!("🎯 Real-Time Video Detection with Multi-Tracker Support\n");
    io::stderr().flush()?;

    // Open video source
    eprint!("📹 Opening video source: {}... ", video_source);
    io::stderr().flush()?;
    let mut cap = if let Ok(cam_id) = video_source.parse::<i32>() {
        VideoCapture::new(cam_id, videoio::CAP_ANY)?
    } else {
        VideoCapture::from_file(&video_source, videoio::CAP_ANY)?
    };

    if !cap.is_opened()? {
        eprintln!("❌ Failed to open video source");
        io::stderr().flush()?;
        return Ok(());
    }

    let fps = cap.get(videoio::CAP_PROP_FPS).unwrap_or(24.0);

    // Try to set FPS for camera sources (doesn't work for video files)
    // For video files, we'll pace with sleep() in the main loop
    if video_source.parse::<i32>().is_ok() {
        cap.set(videoio::CAP_PROP_FPS, fps)?;
    }

    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT).unwrap_or(0.0) as i32;

    eprintln!("✓");
    eprintln!("  • FPS: {:.1}", fps);
    if total_frames > 0 {
        eprintln!("  • Total frames: {}", total_frames);
        eprintln!("  • Duration: {:.1}s", total_frames as f64 / fps);
    }
    io::stderr().flush()?;

    // Create batch executor (replaces detector pool)
    eprintln!("\n📦 Creating batch executor...");
    io::stderr().flush()?;
    let detector_type = DetectorType::RTDETR;
    let batch_size = 1; // No batching possible with 400ms processing time and max_pending=2

    // Read model directory from environment variable or use default
    let model_dir = env::var("DEFENITY_MODEL_DIR").unwrap_or_else(|_| "../models".to_string());

    let detector_config = DetectorConfig {
        model_path: format!("{}/{}", model_dir, model_variant.filename()),
        confidence_threshold,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: false, // Use CPU instead of GPU/Metal
        ..Default::default()
    };

    let executor_config = ExecutorConfig {
        max_queue_depth: 1, // Drop frames if more than 2 pending (backpressure control)
    };
    let max_queue_depth = executor_config.max_queue_depth; // Save before move

    let frame_executor = Arc::new(FrameExecutor::new(
        detector_type,
        detector_config,
        executor_config,
    )?);

    // Create detection pipeline
    let pipeline_config = PipelineConfig {
        overlap: 32,
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
                max_age: 20,        // frames to keep track alive without detections (30*20ms = 600ms)
                min_hits: 1,        // REDUCED from 3 to test track position updates
                iou_threshold: 0.3, // REDUCED from 0.3 to be more permissive for association
                init_tracker_min_score: 0.25, // minimum confidence to create new track (25% - standard value)
                measurement_noise: [1.0, 1.0, 10.0, 10.0], // measurement noise covariance
                process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001], // standard process noise
            })
        }
        TrackingMethod::ByteTrack => {
            TrackingConfig::ByteTrack(ByteTrackConfig {
                max_age: 20,                               // frames to keep track alive (30*20ms = 600ms)
                min_hits: 1,        // REDUCED from 3 to test track position updates
                iou_threshold: 0.3, // IoU threshold for association (standard value)
                init_tracker_min_score: 0.25, // minimum confidence to create new track (25% - standard value)
                high_score_threshold: 0.5,    // high confidence threshold
                low_score_threshold: 0.1,     // low confidence threshold
                measurement_noise: [1.0, 1.0, 10.0, 10.0], // measurement noise
                process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001], // process noise
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

    eprintln!("  ✓ Pipeline ready (with {} tracker)", tracking_method);
    eprintln!("\n💡 Configuration:");
    eprintln!("  • Detector: {} (frame executor)", model_variant.name());
    eprintln!("  • Batch size: {}", batch_size);
    eprintln!("  • Queue depth: {} (backpressure)", max_queue_depth);
    eprintln!("  • Confidence: {:.0}%", confidence_threshold * 100.0);
    eprintln!("  • Classes: {:?}", allowed_classes);
    eprintln!("  • Tracker: {}", tracking_method);
    eprintln!("  • Output FPS: {:.1} (matching input)", fps);
    eprintln!("  • Max latency for extrapolation: 500ms");
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

    let mut frame_id = 0_u64;
    let mut paused = false;

    // Create clones of Arc for threads - these will all share the same atomic
    let stats_processed = Arc::new(AtomicU32::new(0));
    let stats_extrapolated = Arc::new(AtomicU32::new(0));
    let stats_latency_sum = Arc::new(AtomicU32::new(0)); // Store as integer (sum of ms)
    let stats_start = Instant::now();

    eprintln!(
        "📝 Detection strategy: Frame executor self-regulates backpressure (queue depth: {})",
        max_queue_depth
    );
    eprintln!("📝 Starting async capture thread at {:.1} FPS\n", fps);
    io::stderr().flush()?;

    // Create async capture thread - captures frames at video's native FPS
    // TWO separate queues:
    // 1. detector_rx: Can drop frames (non-blocking), used for detection - carries timestamp
    // 2. output_rx: Must process ALL frames (blocking), used for output
    let (detector_tx, detector_rx) = sync_channel::<(Arc<RgbImage>, Instant)>(3); // Detector queue with Arc to avoid cloning
    let (output_tx, output_rx) = sync_channel::<(Arc<RgbImage>, Instant)>(3); // Output queue with Arc to avoid cloning
    let (writer_tx, writer_rx) = sync_channel::<(Mat, u32, u32)>(100);
    let (display_tx, display_rx) = sync_channel::<(Mat, u32, u32)>(3);

    let capture_frame_duration = Duration::from_secs_f64(1.0 / fps as f64);

    let capture_handle = thread::spawn(move || {
        let mut cp_frame_id = 0_u64;
        loop {
            // Check for shutdown signal (non-blocking)
            if let Ok(_) = shutdown_rx_capture.try_recv() {
                log::debug!("Capture thread received shutdown signal");
                break;
            }

            let frame_start = Instant::now();
            cp_frame_id += 1;

            // Pace to target FPS - sleep for remaining time in this frame period
            let elapsed = frame_start.elapsed();
            if elapsed < capture_frame_duration {
                std::thread::sleep(capture_frame_duration - elapsed);
            }

            match capture_frame(&mut cap) {
                Ok(frame) => {
                    let capture_time = Instant::now();
                    let frame_arc = Arc::new(frame); // Wrap in Arc for zero-copy sharing

                    // Send to output queue (NON-BLOCKING - drop if queue full)
                    let _ = output_tx.try_send((Arc::clone(&frame_arc), capture_time));

                    // Send to detector queue (NON-BLOCKING - drop if queue full)
                    let _ = detector_tx.try_send((frame_arc, capture_time));

                    log::debug!("Dispatching frame {} for processing", cp_frame_id);
                }
                Err(_) => {
                    // End of video - send termination signal
                    log::debug!("Capture thread: End of video");
                    let _ = shutdown_tx_display.send(()); // Signal main loop thread to shutdown
                    break;
                }
            }
        }

        // Ensure channels are closed on shutdown
        log::debug!("Capture thread: Closing channels");
    });

    let video_pipeline_for_detector = Arc::clone(&video_pipeline);

    let detector_handle = thread::spawn(move || {
        let mut submitted_frame_id = 0_u64;
        let mut input_frame_id = 0_u64;

        loop {
            // Check for shutdown signal (non-blocking)
            if let Ok(_) = shutdown_rx_detector.try_recv() {
                log::debug!("Detector thread received shutdown signal");
                // Shutdown the video pipeline
                video_pipeline_for_detector.shutdown();
                break;
            }

            let frame_start = Instant::now();

            // Drain DETECTOR queue to get only the LATEST frame
            // This naturally skips frames - if detection is slow, many frames accumulate and we only take the newest
            let mut latest_detector_frame: Option<(Arc<RgbImage>, Instant)> = None;
            let mut frames_drained = 0;

            while let Ok(frame_with_timestamp) = detector_rx.try_recv() {
                // If we already have a frame, the previous one becomes a dropped frame
                if latest_detector_frame.is_some() {
                    // Advance tracker for the dropped frame (async, non-blocking)
                    //video_pipeline_for_detector.advance_tracks();
                    log::warn!("Advanced tracker for dropped frame (rt fps missed)");
                }

                latest_detector_frame = Some(frame_with_timestamp);
                frames_drained += 1;
                input_frame_id += 1;
            }

            if frames_drained > 1 {
                log::debug!("Drained {} frames, using latest", frames_drained);
            }
            log::debug!("Consuming frame {} for detect", input_frame_id);

            // Submit latest detector frame if available
            if let Some((detector_frame, _capture_time)) = latest_detector_frame {
                // Convert image to raw RGB data
                let (width, height) = detector_frame.dimensions();
                let raw_data: Vec<u8> = detector_frame.as_ref().clone().into_raw(); // Clone only when converting to raw data

                let frame = Frame {
                    data: Arc::from(raw_data.into_boxed_slice()), // Zero-copy Arc data
                    width,
                    height,
                    sequence: submitted_frame_id,
                };

                // submit_frame is NON-BLOCKING - if pipeline is busy, frame is dropped
                // This provides natural backpressure without pacing
                if video_pipeline_for_detector.submit_frame(frame) {
                    log::debug!("Submitted frame {} for detect", submitted_frame_id);
                    submitted_frame_id += 1;
                } else {
                    //video_pipeline_for_detector.advance_tracks();
                }
            }

            // Pace to target FPS - sleep for remaining time in this frame period
            let elapsed = frame_start.elapsed();
            if elapsed < capture_frame_duration {
                std::thread::sleep(capture_frame_duration - elapsed);
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
            // Check for shutdown signal (non-blocking)
            if let Ok(_) = shutdown_rx_stats.try_recv() {
                log::debug!("Results thread received shutdown signal");
                break;
            }

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

            // Small sleep to avoid busy loop
            std::thread::sleep(Duration::from_millis(10));
        }
    });

    let mut out_frame_id = 0_u64;

    let writer_handle = thread::spawn(move || {
        let mut file_writer: Option<VideoWriter> = None;

        loop {
            if let Ok(_) = shutdown_rx_writer.try_recv() {
                log::debug!("Writer thread received shutdown signal");
                while let Ok((mat, _width, _height)) = writer_rx.try_recv() {
                    let _ = file_writer.as_mut().unwrap().write(&mat).is_ok();
                }
                break;
            }
            match writer_rx.try_recv() {
                Ok((mat, width, height)) => {
                    if let Some(writer) = &mut file_writer {
                        log::trace!("Writing frame to output video file");
                        let _ = writer.write(&mat).is_ok();
                    } else {
                        file_writer = match VideoWriter::new(
                            output_path,
                            VideoWriter::fourcc('a', 'v', 'c', '1').unwrap(),
                            fps,
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
                        let _ = file_writer.as_mut().unwrap().write(&mat).is_ok();
                    }
                }
                Err(_) => {
                    continue;
                }
            }
        }
        if let Some(writer) = &mut file_writer {
            log::info!("Releasing video writer");
            let _ = writer.release().map_err(|e| e.to_string());
            log::debug!("Output video saved: {}", output_path);
        }
    });

    let output_handle = thread::spawn(move || {
        let mut elapsed = stats_start.elapsed().as_secs_f32();
        let mut avg_fps = frame_id as f32 / elapsed;
        let mut processed = stats_processed.load(Ordering::Relaxed);
        let mut extrapolated = stats_extrapolated.load(Ordering::Relaxed);
        let mut total = processed + extrapolated;
        let mut avg_latency = if total > 0 {
            stats_latency_sum.load(Ordering::Relaxed) as f32 / total as f32
        } else {
            0.0
        };

        loop {
            // Check shutdown signal
            if let Ok(_) = shutdown_rx_output.try_recv() {
                log::debug!("Output thread received shutdown signal");
                break;
            }

            // Try to get frame from OUTPUT queue with timeout
            // This allows the loop to check for exit conditions periodically
            // instead of blocking forever waiting for frames
            let new_captured_frame = match output_rx.recv_timeout(Duration::from_millis(2)) {
                Ok((frame_arc, _timestamp)) => Some(frame_arc),
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

            let frame_arc = if let Some(frame_arc) = new_captured_frame {
                log::debug!("Consuming frame: {} for output", out_frame_id);
                out_frame_id += 1;
                frame_arc // Keep as Arc, clone only when needed
            } else {
                log::warn!("No more frames available, ending video processing");
                break;
            };
            let origin_frame_width = frame_arc.width();
            let origin_frame_height = frame_arc.height();

            // === OPTIMIZATION: Pre-compute scale factors ONCE (after first frame) ===
            // These don't change between frames, so calculate once and reuse
            let (scale_x, scale_y) =
                calculate_scale_factors(origin_frame_width, origin_frame_height, 640.0);

            // Get latest tracker predictions directly from video pipeline cache (synchronous)
            // This bypasses the channel and gets predictions immediately from tracker cache
            let latest_predictions = video_pipeline.get_predictions();
            let num_tracks = latest_predictions
                .iter()
                .filter(|d| d.track_id.is_some())
                .count();

            // Annotate current frame with latest tracker predictions (from VideoPipeline with tracking)
            let mut annotated = frame_arc.as_ref().clone(); // Clone only once when we need to annotate

            // PARALLEL: Pre-compute all annotation data (boxes, labels, colors)
            let annotation_data: Vec<_> =
                prepare_annotation_data(&latest_predictions, scale_x, scale_y);

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
                    frame_id,
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
                opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
            );
            let _ = display_tx
                .try_send((display_mat.clone(), frame_arc.width(), frame_arc.height()))
                .is_ok();

            // Write ALL frames to output video (async, non-blocking)
            let _ = writer_tx
                .try_send((display_mat.clone(), frame_arc.width(), frame_arc.height()))
                .is_ok();

            frame_id += 1;

            // Print progress every N frames (based on FPS)
            let progress_interval = (fps as u64).max(24);
            if frame_id % progress_interval == 0 {
                elapsed = stats_start.elapsed().as_secs_f32();
                avg_fps = frame_id as f32 / elapsed;
                processed = stats_processed.load(Ordering::Relaxed);
                extrapolated = stats_extrapolated.load(Ordering::Relaxed);
                total = processed + extrapolated;
                avg_latency = if total > 0 {
                    stats_latency_sum.load(Ordering::Relaxed) as f32 / total as f32
                } else {
                    0.0
                };

                log::debug!(
                    "Frame {}: {:.1} FPS | Tracks: {} | Display frames: {}",
                    frame_id,
                    avg_fps,
                    num_tracks,
                    frame_id,
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
                io::stderr().flush().ok();
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
            frame_id, elapsed, processed, extrapolated
        );
        eprintln!(
            "==============================================================================="
        );

        eprintln!("\n");
        eprintln!("===================================================");
        eprintln!("FINAL STATISTICS:");
        eprintln!("===================================================");
        eprintln!("Total frames displayed: {}", frame_id);
        eprintln!("Duration: {:.1}s", elapsed);
        eprintln!("Average FPS: {:.1}", avg_fps);
        eprintln!("");
        eprintln!("Detection Stats:");
        eprintln!("  RT-DETR runs: {}", processed);
        eprintln!("  Average latency: {:.0}ms", avg_latency);
        eprintln!("===================================================\n");
        let _ = io::stderr().flush().is_ok();
    });

    // Pre-allocate Mat for reuse (optimization)
    let display_initialized = std::cell::Cell::new(false);
    let mut latest_mat = Mat::default();

    loop {
        if let Ok(_) = shutdown_rx_display.try_recv() {
            log::debug!("Main loop received shutdown signal");
            if !headless {
                let _ = highgui::destroy_all_windows().is_ok();
            }
            break;
        }

        log::debug!("Main Loop: Starting iteration, checking for frames...");

        if !headless {
            while let Ok((mat, width, height)) = display_rx.try_recv() {
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
                    let _ =
                        highgui::resize_window("Detection", window_width, window_height).is_ok();
                    display_initialized.set(true);
                }
                latest_mat = mat.clone();
            }
            let _ = highgui::imshow("Detection", &latest_mat).is_ok();

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
        } else {
            // In headless mode, add a small sleep to prevent busy loop
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    // Send shutdown signals to all threads when main loop ends
    log::info!("Main loop ended, shutting down all threads");
    let _ = shutdown_tx_capture.send(()).is_ok() && capture_handle.join().is_ok();
    let _ = shutdown_tx_detector.send(()).is_ok() && detector_handle.join().is_ok();
    let _ = shutdown_tx_stats.send(()).is_ok() && stats_handle.join().is_ok();
    let _ = shutdown_tx_writer.send(()).is_ok() && writer_handle.join().is_ok();
    let _ = shutdown_tx_output.send(()).is_ok() && output_handle.join().is_ok();

    Ok(())
}
