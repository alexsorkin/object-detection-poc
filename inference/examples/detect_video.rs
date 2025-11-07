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
///   cargo run --release --features metal --example detect_video [--confidence %%] [--classes c1,c2,...] <video_path>
///   cargo run --release --features metal --example detect_video 0  # Use webcam
///
/// Examples:
///   cargo run --release --features metal --example detect_video test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- --confidence 35 test_data/airport.mp4
///   cargo run --release --features metal --example detect_video -- --classes 0,2,5,7 test_data/airport.mp4
use image::{Rgb, RgbImage};
use military_target_detector::detector_trait::DetectorType;
use military_target_detector::frame_executor::{ExecutorConfig, FrameExecutor};
use military_target_detector::frame_pipeline::{DetectionPipeline, PipelineConfig};
use military_target_detector::image_utils::{draw_rect, draw_text, generate_class_color};
use military_target_detector::kalman_operator::KalmanOperator;
use military_target_detector::kalman_tracker::KalmanConfig;
use military_target_detector::types::DetectorConfig;
use military_target_detector::video_pipeline::{Frame, VideoPipeline, VideoPipelineConfig};
use opencv::{
    core::Mat,
    highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use std::env;
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

    // Parse confidence threshold and class filter
    let mut confidence_threshold = 0.50; // Default 50%
    let mut allowed_classes: Vec<u32> = vec![0, 2, 3, 4, 7]; // person, car, motorcycle, airplane, truck
    let mut arg_idx = 1;

    // Parse --confidence parameter
    if args.len() > arg_idx && args[arg_idx] == "--confidence" {
        arg_idx += 1;
        if args.len() > arg_idx {
            match args[arg_idx].parse::<f32>() {
                Ok(val) => {
                    confidence_threshold = (val / 100.0).clamp(0.0, 1.0); // Convert % to 0.0-1.0
                }
                Err(_) => {
                    eprintln!("Invalid confidence threshold. Use a number between 0-100");
                    return Ok(());
                }
            }
            arg_idx += 1;
        }
    }

    // Parse --classes parameter
    if args.len() > arg_idx && args[arg_idx] == "--classes" {
        arg_idx += 1;
        if args.len() > arg_idx {
            allowed_classes = args[arg_idx]
                .split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect();
            if allowed_classes.is_empty() {
                eprintln!("Invalid classes. Use comma-separated numbers (e.g., 0,2,5,7)");
                return Ok(());
            }
            arg_idx += 1;
        }
    }

    let video_source = if args.len() > arg_idx {
        args[arg_idx].clone()
    } else {
        // Use default video if no path provided
        eprintln!("ℹ️  No video path provided, using default: test_data/airport.mp4");
        "test_data/airport.mp4".to_string()
    };

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
    let buffer_size = 2; // Tiny buffer - only allow 2 frames queued max
    let video_config = VideoPipelineConfig {
        max_latency_ms: 500,
        kalman_config: Default::default(),
        buffer_size,
    };

    let video_pipeline = VideoPipeline::new(Arc::clone(&pipeline), video_config);

    // Initialize global Kalman operator with maintenance thread
    let kalman_config = KalmanConfig {
        max_age_ms: 1000,             // Keep tracks for 1s (detection interval is ~600ms)
        iou_threshold: 0.10, // Very low threshold - detections 350-400ms apart can drift significantly
        max_centroid_distance: 150.0, // Match if centroids within 150px (for fast-moving objects)
        process_noise_pos: 5.0, // Higher position uncertainty for fast-moving objects
        process_noise_vel: 10.0, // Much higher velocity uncertainty (fast aircraft/vehicles)
        measurement_noise: 5.0, // Higher detector noise tolerance
        initial_covariance: 20.0, // Higher initial uncertainty
    };
    let kalman_operator = KalmanOperator::init(kalman_config);

    eprintln!("  ✓ Pipeline ready (with Kalman operator + maintenance thread)");
    eprintln!("\n💡 Configuration:");
    eprintln!("  • Detector: RT-DETR (frame executor)");
    eprintln!("  • Batch size: {}", batch_size);
    eprintln!("  • Queue depth: {} (backpressure)", max_queue_depth);
    eprintln!("  • Confidence: {:.0}%", confidence_threshold * 100.0);
    eprintln!("  • Classes: {:?}", allowed_classes);
    eprintln!("  • Kalman tracker: 1s track timeout");
    eprintln!("  • Output FPS: {:.1} (matching input)", fps);
    eprintln!("  • Max latency for extrapolation: 500ms");

    eprintln!("\n🎬 Starting video processing...");
    eprintln!("  Press 'q' to quit, 'p' to pause\n");
    io::stderr().flush()?;

    // We'll create the video writer after we get the first frame (to know actual dimensions)
    let output_path = "output_video.mp4";
    let writer_initialized = std::cell::Cell::new(false);
    let mut frame_tx: Option<std::sync::mpsc::SyncSender<Mat>> = None;
    let mut writer_handle: Option<thread::JoinHandle<Result<(), String>>> = None;

    // Create window for display (updated every N frames for performance)
    highgui::named_window("Detection", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("Detection", 1280, 720)?;

    // Display update interval: update window every frame for smooth playback
    let display_interval = 1; // Update display every frame

    let mut frame_id = 0_u64;
    let mut paused = false;
    let mut stats_processed = 0_u32;
    let mut stats_extrapolated = 0_u32;
    let mut stats_latency_sum = 0.0_f32;
    let stats_start = Instant::now();
    let mut last_kalman_update = Instant::now();

    // FPS pacing: Calculate target frame duration to match video's native FPS
    let frame_duration = Duration::from_secs_f64(1.0 / fps as f64);
    let mut next_frame_time = Instant::now();
    let mut last_rgb_image: Option<RgbImage> = None;

    eprintln!(
        "📝 Detection strategy: Frame executor self-regulates backpressure (queue depth: {})",
        max_queue_depth
    );
    eprintln!("📝 Starting async capture thread at {:.1} FPS\n", fps);
    io::stderr().flush()?;

    // Create async capture thread - captures frames at video's native FPS
    let (capture_tx, capture_rx) = sync_channel::<RgbImage>(3); // Small buffer for latest frames
    let (done_tx, done_rx) = sync_channel::<()>(1); // Signal for end of video
    let capture_frame_duration = Duration::from_secs_f64(1.0 / fps as f64);
    let capture_handle = thread::spawn(move || {
        let mut next_capture_time = Instant::now();

        loop {
            // Pace capture to video's FPS
            let now = Instant::now();
            if now < next_capture_time {
                let sleep_duration = next_capture_time - now;
                std::thread::sleep(sleep_duration);
            }
            next_capture_time += capture_frame_duration;

            match capture_frame(&mut cap) {
                Ok(frame) => {
                    // Send frame to main thread (non-blocking)
                    if capture_tx.try_send(frame).is_err() {
                        // Queue full, drop oldest frame (main thread will get latest)
                        log::trace!("Capture queue full, dropping frame");
                    }
                }
                Err(_) => {
                    // End of video - send termination signal
                    eprintln!("\n📹 Capture thread: End of video");
                    io::stderr().flush().ok();
                    let _ = done_tx.send(()); // Signal main thread
                    break;
                }
            }
        }
    });

    loop {
        let frame_start = Instant::now();

        // Try to get latest frame from capture thread
        let mut new_captured_frame: Option<RgbImage> = None;
        let mut frames_available = 0;

        // Drain capture queue to get the latest frame
        while let Ok(frame) = capture_rx.try_recv() {
            new_captured_frame = Some(frame);
            frames_available += 1;
        }

        // Determine what frame to use and whether to send to detector
        let (rgb_image, send_to_detector) = if let Some(captured) = new_captured_frame {
            // Got a new frame from capture thread
            last_rgb_image = Some(captured.clone());

            // Check if we're lagging more than 2 frames behind
            if frames_available > 2 {
                log::warn!("⚠️ Lagging {} frames behind capture", frames_available);
            }

            (captured, true) // New frame - send to detector and output
        } else if let Some(ref last_frame) = last_rgb_image {
            // Check if capture thread signaled end of video
            if done_rx.try_recv().is_ok() {
                eprintln!("\n📹 End of video");
                io::stderr().flush()?;
                break;
            }

            // No new frame available - repeat last frame for output only
            log::debug!("⏭️ No new frame, repeating last frame for output");
            (last_frame.clone(), false) // Repeated frame - output only, skip detector
        } else {
            // No frames captured yet - check if capture thread signaled end
            if done_rx.try_recv().is_ok() {
                eprintln!("\n📹 End of video (no frames captured)");
                io::stderr().flush()?;
                break;
            }
            // Wait a bit for first frame
            std::thread::sleep(Duration::from_millis(10));
            continue;
        };

        // Pace to target FPS
        if frame_start < next_frame_time {
            let sleep_duration = next_frame_time - frame_start;
            std::thread::sleep(sleep_duration);
        }
        next_frame_time += frame_duration;

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

        // Get results from detection pipeline (async - may or may not have results)
        while let Some(result) = video_pipeline.try_get_result() {
            if result.is_extrapolated {
                stats_extrapolated += 1;
            } else {
                stats_processed += 1;

                // Update Kalman operator with real detections
                // Use WALL-CLOCK time, not frame timestamp
                let current_time = Instant::now();
                let dt = current_time
                    .duration_since(last_kalman_update)
                    .as_secs_f32();

                let operator = kalman_operator.lock().unwrap();
                let _ = operator.send_update(result.detections.clone(), dt);
                drop(operator);

                last_kalman_update = current_time;
            }
            stats_latency_sum += result.latency_ms;
        }

        // Submit frame to detector ONLY if it's a newly captured frame (not repeated)
        if send_to_detector {
            let frame = Frame {
                frame_id,
                image: rgb_image.clone(),
                timestamp: frame_start,
            };
            video_pipeline.try_submit_frame(frame);
        } else {
            log::debug!("⏭️  Skipping detector for repeated frame {}", frame_id);
        }

        // Get current Kalman predictions (may be empty if no tracks yet)
        let operator = kalman_operator.lock().unwrap();
        let detections_to_display = operator.get_predictions();
        drop(operator);

        // Annotate current frame with Kalman predictions
        let mut annotated = rgb_image.clone();

        // Calculate scale factors (detections are in resized image coordinates)
        // Pipeline resizes so LONGEST dimension = 640, preserving aspect ratio
        // This ensures the frame fits in a single 640x640 tile
        let target_size = 640.0_f32;
        let tile_size = if orig_width > orig_height {
            target_size.min(orig_width as f32)
        } else {
            target_size.min(orig_height as f32)
        };

        let scale = if orig_width > orig_height {
            tile_size / orig_width as f32
        } else {
            tile_size / orig_height as f32
        };

        let resized_width = (orig_width as f32 * scale) as u32;
        let resized_height = (orig_height as f32 * scale) as u32;

        // Scale from pipeline coords back to current frame coords
        let scale_x = orig_width as f32 / resized_width as f32;
        let scale_y = orig_height as f32 / resized_height as f32;

        // Draw detections with track ID visualization
        for det in &detections_to_display {
            // Use consistent color per class (person=green, car=blue, etc.)
            let color = generate_class_color(det.class_id);

            // Scale detection coordinates back to original image size
            let scaled_x = (det.x * scale_x) as i32;
            let scaled_y = (det.y * scale_y) as i32;
            let scaled_w = (det.w * scale_x) as u32;
            let scaled_h = (det.h * scale_y) as u32;

            draw_rect(
                &mut annotated,
                scaled_x,
                scaled_y,
                scaled_w,
                scaled_h,
                color,
                2,
            );

            // Label with track ID
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

            if scaled_h > 20 && scaled_w > 30 {
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

        // Draw stats overlay
        let operator = kalman_operator.lock().unwrap();
        let num_tracks = operator.get_predictions().len();
        drop(operator);

        let stats_text = format!(
            "Frame: {} | Tracks: {} | Detections: {} RT-DETR",
            frame_id, num_tracks, stats_processed
        );

        draw_text(
            &mut annotated,
            &stats_text,
            10,
            30,
            Rgb([255, 255, 255]),
            Some(Rgb([0, 0, 0])),
        );

        // Convert annotated frame to Mat for display and writing
        let display_width = annotated.width();
        let display_height = annotated.height();

        // Convert back to Mat for display
        let mut display_mat = Mat::default();
        let data_slice = annotated.as_raw();

        // Create Mat from RGB data - opencv expects CV_8UC3 format
        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                display_height as i32,
                display_width as i32,
                opencv::core::CV_8UC3,
                data_slice.as_ptr() as *mut _,
                opencv::core::Mat_AUTO_STEP,
            )?
        };

        imgproc::cvt_color(
            &mat,
            &mut display_mat,
            imgproc::COLOR_RGB2BGR,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Display (every Nth frame to reduce overhead)
        if frame_id % display_interval == 0 {
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
            let total = stats_processed + stats_extrapolated;
            let avg_latency = if total > 0 {
                stats_latency_sum / total as f32
            } else {
                0.0
            };

            let operator = kalman_operator.lock().unwrap();
            let num_tracks = operator.get_predictions().len();
            drop(operator);

            eprintln!(
                "Frame {}: {:.1} FPS | Tracks: {} | Real detections: {} | Avg Latency: {:.0}ms",
                frame_id, avg_fps, num_tracks, stats_processed, avg_latency
            );
            io::stderr().flush().ok();
        }

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

    // Final statistics
    let total_time = stats_start.elapsed().as_secs_f32();
    let total = stats_processed + stats_extrapolated;
    let avg_fps = frame_id as f32 / total_time;
    let avg_latency = if total > 0 {
        stats_latency_sum / total as f32
    } else {
        0.0
    };

    eprintln!("\n📊 Final Statistics:");
    eprintln!("  • Total frames displayed: {}", frame_id);
    eprintln!("  • Duration: {:.1}s", total_time);
    eprintln!("  • Average FPS: {:.1}", avg_fps);
    eprintln!("\n  Detection Stats:");
    eprintln!("    - Real detections: {}", stats_processed);
    eprintln!("    - Average latency: {:.0}ms", avg_latency);
    io::stderr().flush()?;

    // Close video writer channel and wait for thread to finish
    drop(frame_tx);
    if let Some(handle) = writer_handle {
        if let Err(e) = handle.join() {
            eprintln!("⚠️  Writer thread error: {:?}", e);
        }
    }

    // Wait for capture thread to finish
    if let Err(e) = capture_handle.join() {
        eprintln!("⚠️  Capture thread error: {:?}", e);
    }

    eprintln!("\n✅ Output video saved: {}", output_path);
    io::stderr().flush()?;

    highgui::destroy_all_windows()?;
    Ok(())
}
