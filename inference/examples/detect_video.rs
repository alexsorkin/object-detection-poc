/// Real-time Video Detection with Kalman Filter Extrapolation
///
/// Processes video files or camera streams with temporal tracking:
/// - Runs detection pipeline at processing FPS (5-10 FPS based on GPU)
/// - Maintains 30 FPS output using Kalman filter extrapolation
/// - Automatically switches to extrapolation when latency > 500ms
///
/// Visual indicators:
/// - GREEN boxes: Real detections from neural network
/// - YELLOW boxes: Kalman filter extrapolations
///
/// Usage:
///   cargo run --release --features metal --example detect_video [OPTIONS] <video_path>
///
/// Options (can be in any order):
///   --confidence <0-100>    Detection confidence threshold (default: 50)
///   --classes <id,id,...>   Comma-separated class IDs to detect (default: 0,2,3,4,7)
///   --headless              Run without display window (default: show window)
///
/// Examples:
///   cargo run --release --features metal --example detect_video test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- --confidence 35 test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- test_data/airport.mp4 --confidence 60 --headless
///   cargo run --release --features metal --example detect_video -- --headless --classes 0,2,5,7 test_data/airport.mp4
///   cargo run --release --features metal --example detect_video 0  # Use webcam
use image::{Rgb, RgbImage};
use military_target_detector::detector_trait::DetectorType;
use military_target_detector::frame_executor::{ExecutorConfig, FrameExecutor};
use military_target_detector::frame_pipeline::{DetectionPipeline, PipelineConfig};
use military_target_detector::image_utils::{draw_rect, draw_text, generate_class_color};
use military_target_detector::kalman_tracker::KalmanConfig;
use military_target_detector::types::DetectorConfig;
use military_target_detector::video_pipeline::{Frame, VideoPipeline, VideoPipelineConfig};
use opencv::{
    core::Mat,
    highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use rayon::prelude::*;
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

    eprintln!("🎯 Real-Time Video Detection with Kalman Extrapolation\n");
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

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(1920.0) as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(1080.0) as i32;
    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT).unwrap_or(0.0) as i32;

    eprintln!("✓");
    eprintln!("  • Resolution: {}x{}", frame_width, frame_height);
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

    let detector_config = DetectorConfig {
        fp16_model_path: Some("../models/rtdetr_v2_r18vd_batch.onnx".to_string()),
        fp32_model_path: Some("../models/rtdetr_v2_r18vd_batch.onnx".to_string()),
        confidence_threshold,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: true,
        ..Default::default()
    };

    let executor_config = ExecutorConfig {
        max_queue_depth: 2, // Drop frames if more than 2 pending (backpressure control)
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

    // Create video pipeline with Kalman tracker
    let buffer_size = 10; // Tiny buffer - only allow 2 frames queued max
    let video_config = VideoPipelineConfig {
        max_latency_ms: 500,
        kalman_config: KalmanConfig {
            max_age_ms: 500,              // Keep tracks for 1s (detection interval is ~600ms)
            iou_threshold: 0.10, // Very low threshold - detections 350-400ms apart can drift significantly
            max_centroid_distance: 150.0, // Match if centroids within 150px (for fast-moving objects)
            process_noise_pos: 5.0,       // Higher position uncertainty for fast-moving objects
            process_noise_vel: 10.0, // Much higher velocity uncertainty (fast aircraft/vehicles)
            measurement_noise: 5.0,  // Higher detector noise tolerance
            initial_covariance: 20.0, // Higher initial uncertainty
        },
        buffer_size,
    };

    let video_pipeline = Arc::new(VideoPipeline::new(Arc::clone(&pipeline), video_config));

    eprintln!("  ✓ Pipeline ready (with internal Kalman tracker)");
    eprintln!("\n💡 Configuration:");
    eprintln!("  • Detector: RT-DETR (frame executor)");
    eprintln!("  • Batch size: {}", batch_size);
    eprintln!("  • Queue depth: {} (backpressure)", max_queue_depth);
    eprintln!("  • Confidence: {:.0}%", confidence_threshold * 100.0);
    eprintln!("  • Classes: {:?}", allowed_classes);
    eprintln!("  • Kalman tracker: 1s track timeout");
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
    let writer_initialized = std::cell::Cell::new(false);
    let mut frame_tx: Option<std::sync::mpsc::SyncSender<Mat>> = None;
    let mut writer_handle: Option<thread::JoinHandle<Result<(), String>>> = None;

    // Create window for display - will be resized after first frame to match video size (max 640p)
    if !headless {
        highgui::named_window("Detection", highgui::WINDOW_NORMAL)?;
    }

    // Display update interval: update window every frame for smooth playback
    let display_interval = 1; // Update display every frame

    let mut frame_id = 0_u64;
    let mut paused = false;

    // Thread-safe stats for detector thread
    let stats_processed = Arc::new(AtomicU32::new(0));
    let stats_extrapolated = Arc::new(AtomicU32::new(0));
    let stats_latency_sum = Arc::new(AtomicU32::new(0)); // Store as integer (sum of ms)
    let stats_start = Instant::now();

    // Track latest detections from VideoPipeline (includes Kalman tracking)
    let mut latest_detections: Vec<military_target_detector::frame_pipeline::TileDetection> =
        Vec::new();
    let mut num_tracks = 0;

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
    let (detector_tx, detector_rx) = sync_channel::<(RgbImage, Instant)>(3); // Detector queue with timestamp
    let (output_tx, output_rx) = sync_channel::<(RgbImage, Instant)>(3); // Output queue - must process all
    let (output_done_tx, output_done_rx) = sync_channel::<()>(1); // Signal for end of video
    let (results_done_tx, results_done_rx) = sync_channel::<()>(1); // Signal for end of video
    let (detector_done_tx, detector_done_rx) = sync_channel::<()>(1); // Signal for end of video
    let capture_frame_duration = Duration::from_secs_f64(1.0 / fps as f64);

    let capture_handle = thread::spawn(move || {
        let mut cp_frame_id = 0_u64;
        loop {
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

                    // Send to output queue (NON-BLOCKING - drop if queue full)
                    let _ = output_tx.try_send((frame.clone(), capture_time));

                    // Send to detector queue (NON-BLOCKING - drop if queue full)
                    let _ = detector_tx.try_send((frame.clone(), capture_time));

                    log::debug!("Dispatching frame {} for processing", cp_frame_id);
                }
                Err(_) => {
                    // End of video - send termination signal
                    log::debug!("Capture thread: End of video");
                    let _ = output_done_tx.send(()); // Signal main thread
                    let _ = detector_done_tx.send(()); // Signal main thread
                    let _ = results_done_tx.send(()); // Signal main thread
                                                      // Explicitly drop senders to close channels
                    drop(detector_tx);
                    drop(output_tx);
                    break;
                }
            }
        }
    });

    let video_pipeline_for_detector = Arc::clone(&video_pipeline);

    let detector_handle = thread::spawn(move || {
        let mut submitted_frame_id = 0_u64;
        let mut input_frame_id = 0_u64;

        loop {
            let frame_start = Instant::now();

            if let Ok(_) = detector_done_rx.try_recv() {
                log::debug!("Detector thread: End of video");
                break;
            }

            // Drain DETECTOR queue to get only the LATEST frame
            // This naturally skips frames - if detection is slow, many frames accumulate and we only take the newest
            let mut latest_detector_frame: Option<(RgbImage, Instant)> = None;
            let mut frames_drained = 0;
            while let Ok(frame_with_timestamp) = detector_rx.try_recv() {
                latest_detector_frame = Some(frame_with_timestamp);
                frames_drained += 1;
                input_frame_id += 1;
            }

            if frames_drained > 1 {
                log::debug!("Drained {} frames, using latest", frames_drained);
            }
            log::debug!("Consuming frame {} for detect", input_frame_id);

            // Submit latest detector frame if available
            if let Some((detector_frame, capture_time)) = latest_detector_frame {
                let frame = Frame {
                    frame_id: submitted_frame_id,
                    image: detector_frame,
                    timestamp: capture_time,
                };

                // try_submit_frame is NON-BLOCKING - if pipeline is busy, frame is dropped
                // This provides natural backpressure without pacing
                if video_pipeline_for_detector.try_submit_frame(frame) {
                    log::debug!("Submitted frame {} for detect", submitted_frame_id);
                    submitted_frame_id += 1;
                } else {
                    log::warn!("Pipeline busy, frame dropped");
                }
            }

            // Pace to target FPS - sleep for remaining time in this frame period
            let elapsed = frame_start.elapsed();
            if elapsed < capture_frame_duration {
                std::thread::sleep(capture_frame_duration - elapsed);
            }
        }
    });

    // Create result collector thread - consumes detection results asynchronously
    // This prevents result retrieval from blocking the main display loop
    let video_pipeline_for_results = Arc::clone(&video_pipeline);
    let stats_processed_for_results = Arc::clone(&stats_processed);
    let stats_extrapolated_for_results = Arc::clone(&stats_extrapolated);
    let stats_latency_for_results = Arc::clone(&stats_latency_sum);

    // Channel to send latest detections to main thread for display
    let (detections_tx, detections_rx) =
        sync_channel::<Vec<military_target_detector::frame_pipeline::TileDetection>>(2);

    let results_handle = thread::spawn(move || {
        loop {
            if let Ok(_) = results_done_rx.try_recv() {
                log::debug!("Results thread: End of video");
                break;
            }

            // Non-blocking check for results
            if let Some(result) = video_pipeline_for_results.try_get_result() {
                // Update stats
                if result.is_extrapolated {
                    stats_extrapolated_for_results.fetch_add(1, Ordering::Relaxed);
                } else {
                    stats_processed_for_results.fetch_add(1, Ordering::Relaxed);
                }
                stats_latency_for_results.fetch_add(result.latency_ms as u32, Ordering::Relaxed);

                // Send detections to main thread (non-blocking)
                let _ = detections_tx.try_send(result.detections);
            }

            // Small sleep to avoid busy loop
            std::thread::sleep(Duration::from_millis(5));
        }
    });

    let mut out_frame_id = 0_u64;

    // Pre-allocate Mat for reuse (optimization)
    let mut display_mat = Mat::default();

    // === OPTIMIZATION: Pre-compute color palette for all 80 COCO classes ===
    let color_palette: Vec<Rgb<u8>> = (0..80)
        .map(|class_id| generate_class_color(class_id))
        .collect();

    loop {
        if let Ok(_) = output_done_rx.try_recv() {
            log::debug!("Main Loop: End of video (no frames captured)");
            break;
        }

        // Try to get frame from OUTPUT queue with timeout
        // This allows the loop to check for exit conditions periodically
        // instead of blocking forever waiting for frames
        let new_captured_frame = match output_rx.recv_timeout(Duration::from_millis(5)) {
            Ok((frame, _timestamp)) => Some(frame),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // No frame yet, continue loop to check exit conditions
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // Channel closed - capture thread finished
                log::info!("Output channel closed, ending video processing");
                break;
            }
        };

        let rgb_image = if let Some(frame) = new_captured_frame {
            log::debug!("Consuming frame: {} for output", out_frame_id);
            out_frame_id += 1;
            frame
        } else {
            log::warn!("No more frames available, ending video processing");
            break;
        };

        // Initialize video writer on first frame (now we know actual dimensions)
        let orig_width = rgb_image.width();
        let orig_height = rgb_image.height();

        if !writer_initialized.get() {
            let fourcc = VideoWriter::fourcc('a', 'v', 'c', '1')?;
            let mut new_writer = VideoWriter::new(
                output_path,
                fourcc,
                fps,
                opencv::core::Size::new(orig_width as i32, orig_height as i32),
                true,
            )?;

            if !new_writer.is_opened()? {
                eprintln!("❌ Failed to open video writer");
                io::stderr().flush()?;
                return Ok(());
            }

            eprintln!(
                "  ✓ Writing output to: {} ({}x{})\n",
                output_path, orig_width, orig_height
            );
            io::stderr().flush()?;

            // Resize display window to match video size (capped at 640p)
            if !headless {
                let max_height = 640;
                let (window_width, window_height) = if orig_height > max_height {
                    // Scale down to 640p preserving aspect ratio
                    let scale = max_height as f32 / orig_height as f32;
                    ((orig_width as f32 * scale) as i32, max_height as i32)
                } else {
                    // Use native resolution
                    (orig_width as i32, orig_height as i32)
                };
                highgui::resize_window("Detection", window_width, window_height)?;
            }

            // Create async video writer thread with large buffer
            let (tx, rx) = sync_channel::<Mat>(120); // Buffer ~5 seconds at 24fps
            let handle = thread::spawn(move || -> Result<(), String> {
                for mat in rx {
                    new_writer.write(&mat).map_err(|e| e.to_string())?;
                }
                new_writer.release().map_err(|e| e.to_string())?;
                Ok(())
            });

            frame_tx = Some(tx);
            writer_initialized.set(true);
            writer_handle = Some(handle);
        }

        // === OPTIMIZATION: Pre-compute scale factors ONCE (after first frame) ===
        // These don't change between frames, so calculate once and reuse
        let (scale_x, scale_y) = {
            let target_size = 640.0_f32;
            let scale = if orig_width > orig_height {
                (target_size / orig_width as f32).min(1.0)
            } else {
                (target_size / orig_height as f32).min(1.0)
            };

            let resized_width = (orig_width as f32 * scale) as u32;
            let resized_height = (orig_height as f32 * scale) as u32;

            (
                orig_width as f32 / resized_width as f32,
                orig_height as f32 / resized_height as f32,
            )
        };

        // Get latest detections from results thread (non-blocking)
        if let Ok(detections) = detections_rx.try_recv() {
            latest_detections = detections;
            num_tracks = latest_detections
                .iter()
                .filter(|d| d.track_id.is_some())
                .count();
        }

        // Annotate current frame with latest detections (from VideoPipeline with Kalman)
        let mut annotated = rgb_image.clone();

        // PARALLEL: Pre-compute all annotation data (boxes, labels, colors)
        // Uses pre-computed scale_x, scale_y and color_palette
        let annotation_data: Vec<_> = latest_detections
            .par_iter()
            .map(|det| {
                let color = color_palette[det.class_id as usize]; // Lookup, not compute
                let scaled_x = (det.x * scale_x) as i32;
                let scaled_y = (det.y * scale_y) as i32;
                let scaled_w = (det.w * scale_x) as u32;
                let scaled_h = (det.h * scale_y) as u32;

                let label = if let Some(track_id) = det.track_id {
                    format!(
                        "#{} {} {:.0}%",
                        track_id,
                        det.class_name,
                        det.confidence * 100.0
                    )
                } else {
                    format!("{} {:.0}%", det.class_name, det.confidence * 100.0)
                };

                let show_label = scaled_h > 20 && scaled_w > 30;

                (
                    scaled_x, scaled_y, scaled_w, scaled_h, color, label, show_label,
                )
            })
            .collect();

        // SEQUENTIAL: Apply annotations to image (image operations are not thread-safe)
        for (scaled_x, scaled_y, scaled_w, scaled_h, color, label, show_label) in annotation_data {
            draw_rect(
                &mut annotated,
                scaled_x,
                scaled_y,
                scaled_w,
                scaled_h,
                color,
                2,
            );

            if show_label {
                draw_text(
                    &mut annotated,
                    &label,
                    scaled_x + 3,
                    scaled_y + 15,
                    Rgb([255, 255, 255]),
                    None,
                );
            }
        }

        // Draw stats overlay (only when displaying to reduce overhead)
        if frame_id % display_interval == 0 {
            let stats_text = format!(
                "Frame: {} | Tracks: {} | Detections: {} RT-DETR",
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
            )?
        };

        // Reuse pre-allocated display_mat buffer for BGR conversion
        imgproc::cvt_color(
            &mat,
            &mut display_mat,
            imgproc::COLOR_RGB2BGR,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Display (every Nth frame to reduce overhead)
        if !headless && frame_id % display_interval == 0 {
            highgui::imshow("Detection", &display_mat)?;
        }

        // Write ALL frames to output video (async, non-blocking)
        if let Some(ref tx) = frame_tx {
            // Use try_send to avoid blocking the main loop
            // If queue is full, skip this frame (writer can't keep up)
            let _ = tx.try_send(display_mat.clone());
        }

        frame_id += 1;

        // Print progress every N frames (based on FPS)
        let progress_interval = (fps as u64).max(24);
        if frame_id % progress_interval == 0 {
            let elapsed = stats_start.elapsed().as_secs_f32();
            let avg_fps = frame_id as f32 / elapsed;
            let processed = stats_processed.load(Ordering::Relaxed);
            let extrapolated = stats_extrapolated.load(Ordering::Relaxed);
            let total = processed + extrapolated;
            let avg_latency = if total > 0 {
                stats_latency_sum.load(Ordering::Relaxed) as f32 / total as f32
            } else {
                0.0
            };

            log::debug!(
                "Frame {}: {:.1} FPS | Tracks: {}",
                frame_id,
                avg_fps,
                num_tracks,
            );
            log::debug!(
                "Real detections: {} | Avg Latency: {:.0}ms",
                processed,
                avg_latency
            );
            io::stderr().flush().ok();
        }

        // Handle keyboard input with minimal delay (1ms for UI responsiveness)
        if !headless {
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

    let total_time = stats_start.elapsed().as_secs_f32();
    let processed = stats_processed.load(Ordering::Relaxed);
    let extrapolated = stats_extrapolated.load(Ordering::Relaxed);
    let total = processed + extrapolated;
    let avg_fps = frame_id as f32 / total_time;
    let avg_latency = if total > 0 {
        stats_latency_sum.load(Ordering::Relaxed) as f32 / total as f32
    } else {
        0.0
    };

    // Close video writer channel and wait for thread to finish
    drop(frame_tx);
    if let Some(handle) = writer_handle {
        match handle.join() {
            Ok(Ok(())) => {
                log::debug!("Video writer finished successfully");
            }
            Ok(Err(e)) => {
                eprintln!("⚠️  Writer thread error: {}", e);
                io::stderr().flush()?;
            }
            Err(e) => {
                eprintln!("⚠️  Writer thread panicked: {:?}", e);
                io::stderr().flush()?;
            }
        }
    }

    eprintln!("\n🧹 Waiting for threads to finish...");

    // Wait for threads to finish
    if let Err(e) = capture_handle.join() {
        eprintln!("⚠️  Capture thread error: {:?}", e);
    } else {
        eprintln!("  ✓ Capture thread finished");
    }
    if let Err(e) = detector_handle.join() {
        eprintln!("⚠️  Detector thread error: {:?}", e);
    } else {
        eprintln!("  ✓ Detector thread finished");
    }
    if let Err(e) = results_handle.join() {
        eprintln!("⚠️  Results thread error: {:?}", e);
    } else {
        eprintln!("  ✓ Results thread finished");
    }
    io::stderr().flush()?;

    log::debug!("Output video saved: {}", output_path);

    eprintln!("\n");
    eprintln!("===============================================================================");
    eprintln!(
        "Finished with: frame_id={}, total_time={}, processed={}, extrapolated={}",
        frame_id, total_time, processed, extrapolated
    );
    eprintln!("===============================================================================");

    eprintln!("\n");
    eprintln!("===================================================");
    eprintln!("FINAL STATISTICS:");
    eprintln!("===================================================");
    eprintln!("Total frames displayed: {}", frame_id);
    eprintln!("Duration: {:.1}s", total_time);
    eprintln!("Average FPS: {:.1}", avg_fps);
    eprintln!("");
    eprintln!("Detection Stats:");
    eprintln!("  Real detections: {}", processed);
    eprintln!("  Average latency: {:.0}ms", avg_latency);
    eprintln!("===================================================\n");
    io::stderr().flush()?;

    if !headless {
        highgui::destroy_all_windows()?;
    }
    Ok(())
}
