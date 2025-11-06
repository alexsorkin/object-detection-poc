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
use military_target_detector::batch_executor::BatchConfig;
use military_target_detector::detector_pool::DetectorPool;
use military_target_detector::detector_trait::DetectorType;
use military_target_detector::image_utils::{draw_rect, draw_text, generate_class_color};
use military_target_detector::kalman_tracker::{KalmanConfig, MultiObjectTracker};
use military_target_detector::pipeline::{DetectionPipeline, PipelineConfig};
use military_target_detector::realtime_pipeline::{
    Frame, FrameResult, RealtimePipeline, RealtimePipelineConfig,
};
use military_target_detector::types::DetectorConfig;
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
use std::time::Instant;

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
    let mut allowed_classes: Vec<u32> = vec![0, 2, 3, 4, 5, 7]; // person, car, motorcycle, airplane, bus, truck
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
        eprintln!(
            "Usage: {} [--confidence %%] [--classes c1,c2,...] <video_path | camera_id>",
            args[0]
        );
        eprintln!("  --confidence: Detection confidence threshold (0-100, default: 50)");
        eprintln!("  --classes:    Comma-separated class IDs to detect (default: 0,2,3,4,5,7)");
        eprintln!("                0=person, 2=car, 3=motorcycle, 4=airplane, 5=bus, 7=truck");
        eprintln!("  video_path:   Path to video file");
        eprintln!("  camera_id:    0 for default webcam");
        eprintln!("\nExamples:");
        eprintln!("  {} test_data/airport.mp4", args[0]);
        eprintln!("  {} --confidence 35 test_data/airport.mp4", args[0]);
        eprintln!("  {} --classes 0,2,5,7 test_data/airport.mp4", args[0]);
        return Ok(());
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

    // Create detector pool
    eprintln!("\n📦 Creating detector pool...");
    io::stderr().flush()?;
    let detector_type = DetectorType::RTDETR;
    let num_workers = 4;
    let batch_size = 2;

    let detector_config = DetectorConfig {
        fp16_model_path: Some("../models/rtdetr_v2_r18vd_batch.onnx".to_string()),
        fp32_model_path: Some("../models/rtdetr_v2_r18vd_batch.onnx".to_string()),
        confidence_threshold,
        nms_threshold: 0.45,
        input_size: (640, 640),
        use_gpu: true,
        ..Default::default()
    };

    let batch_config = BatchConfig {
        batch_size,
        timeout_ms: 50,
    };

    let detector_pool = Arc::new(DetectorPool::new(
        num_workers,
        detector_type,
        detector_config,
        batch_config,
    )?);

    // Create detection pipeline
    let pipeline_config = PipelineConfig {
        overlap: 32,
        allowed_classes: allowed_classes.clone(),
        iou_threshold: 0.5,
    };

    let pipeline = Arc::new(DetectionPipeline::new(
        Arc::clone(&detector_pool),
        pipeline_config,
    ));

    // Create real-time pipeline with Kalman tracker
    let buffer_size = 2; // Tiny buffer - only allow 2 frames queued max
    let rt_config = RealtimePipelineConfig {
        max_latency_ms: 500,
        kalman_config: Default::default(),
        buffer_size,
    };

    let rt_pipeline = RealtimePipeline::new(Arc::clone(&pipeline), rt_config);

    // Create single shared Kalman tracker (simple, synchronous approach)
    let mut kalman_tracker = MultiObjectTracker::new(KalmanConfig::default());

    eprintln!("  ✓ Pipeline ready");
    eprintln!("\n💡 Configuration:");
    eprintln!("  • Detector: RT-DETR");
    eprintln!("  • Workers: {}", num_workers);
    eprintln!("  • Batch size: {}", batch_size);
    eprintln!("  • Confidence: {:.0}%", confidence_threshold * 100.0);
    eprintln!("  • Classes: {:?}", allowed_classes);
    eprintln!("  • Kalman tracker: Single synchronous tracker");
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

    // Display update interval: update window every N frames to maintain real-time speed
    let display_interval = 5; // Update display every 5 frames

    let mut frame_id = 0_u64;
    let mut paused = false;
    let mut stats_processed = 0_u32;
    let mut stats_extrapolated = 0_u32;
    let mut stats_latency_sum = 0.0_f32;
    let mut stats_displayed_fresh = 0_u32; // Frames displayed with fresh/recent result
    let mut stats_displayed_extrapolated = 0_u32; // Frames displayed with old extrapolated result
    let stats_start = Instant::now();

    // Strategy: Only submit if result queue is not backed up
    // This prevents overwhelming the pipeline and allows Kalman to work
    let mut latest_result: Option<FrameResult> = None;
    let mut latest_result_time = Instant::now();
    let max_pending = 2; // Only allow 2 frames in flight at a time
    let mut pending_count = 0_usize;
    let mut last_kalman_update = Instant::now();

    eprintln!(
        "📝 Simple strategy: RT-DETR every 40th frame, Kalman predictions for smooth output\n",
    );
    io::stderr().flush()?;

    loop {
        let frame_start = Instant::now();

        // Capture frame
        let mut mat_frame = Mat::default();
        if !cap.read(&mut mat_frame)? || mat_frame.empty() {
            eprintln!("\n📹 End of video");
            io::stderr().flush()?;
            break;
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
        let orig_width = rgb_mat.cols() as u32;
        let orig_height = rgb_mat.rows() as u32;
        let data = rgb_mat.data_bytes()?.to_vec();
        let rgb_image =
            RgbImage::from_vec(orig_width, orig_height, data).ok_or("Failed to create RgbImage")?;

        // Initialize video writer on first frame (now we know actual dimensions)
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

            // Create async video writer thread
            let (tx, rx) = sync_channel::<Mat>(30);
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

        // Get results from detection pipeline
        while let Some(result) = rt_pipeline.try_get_result() {
            if result.is_extrapolated {
                stats_extrapolated += 1;
            } else {
                stats_processed += 1;

                // Update Kalman tracker with real detections (synchronous)
                let dt = frame_start.duration_since(last_kalman_update).as_secs_f32();
                kalman_tracker.update(&result.detections, dt);
                last_kalman_update = frame_start;
            }
            stats_latency_sum += result.latency_ms;

            latest_result = Some(result.clone());
            latest_result_time = result.timestamp;
            pending_count = pending_count.saturating_sub(1);
        }

        // Predict Kalman tracker forward for current frame (even without new detections)
        let dt = frame_start.duration_since(last_kalman_update).as_secs_f32();
        if dt > 0.001 {
            // Only predict if time has passed
            kalman_tracker.update(&[], dt); // Empty detections = prediction only
            last_kalman_update = frame_start;
        }

        // Submit detection for every 40th frame (RT-DETR on CPU is ~0.3-0.4 FPS)
        let should_detect = frame_id % 40 == 0;

        if should_detect && pending_count < max_pending {
            let frame = Frame {
                frame_id,
                image: rgb_image.clone(),
                timestamp: frame_start,
            };
            if rt_pipeline.try_submit_frame(frame) {
                pending_count += 1;
            }
        }

        // Get current Kalman predictions for display (synchronous read)
        let kalman_predictions = kalman_tracker.get_predictions();

        // Annotate current frame with Kalman predictions
        let mut annotated = rgb_image.clone();

        // Use Kalman predictions if available, otherwise fall back to latest result
        let empty_vec = Vec::new();
        let detections_to_display: &Vec<_> = if !kalman_predictions.is_empty() {
            stats_extrapolated += 1;
            stats_displayed_extrapolated += 1;
            &kalman_predictions
        } else if let Some(ref result) = latest_result {
            // Track display stats: fresh result vs extrapolated
            let dt = frame_start.duration_since(latest_result_time).as_secs_f32();
            if dt < 0.05 {
                stats_displayed_fresh += 1;
            } else {
                stats_displayed_extrapolated += 1;
            }
            &result.detections
        } else {
            // No detections yet
            &empty_vec
        };

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
        for det in detections_to_display {
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
        let stats_text = format!(
            "Frame: {} | Tracks: {} | Detections: {} RT-DETR",
            frame_id,
            kalman_tracker.num_tracks(),
            stats_processed
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

        // Write to output video asynchronously (non-blocking)
        if let Some(ref tx) = frame_tx {
            // Use try_send to avoid blocking the main loop if video writer is slow
            if let Err(std::sync::mpsc::TrySendError::Full(_)) = tx.try_send(display_mat.clone()) {
                // Drop frame if video writer can't keep up
                log::warn!("Video writer queue full, dropping frame {}", frame_id);
            }
        }

        frame_id += 1;

        // Print progress every N frames (based on FPS)
        let progress_interval = (fps as u64).max(24);
        if frame_id % progress_interval == 0 {
            let elapsed = stats_start.elapsed().as_secs_f32();
            let avg_fps = frame_id as f32 / elapsed;
            let total_displayed = stats_displayed_fresh + stats_displayed_extrapolated;
            let display_extrapolation_rate = if total_displayed > 0 {
                stats_displayed_extrapolated as f32 / total_displayed as f32 * 100.0
            } else {
                0.0
            };
            let total = stats_processed + stats_extrapolated;
            let avg_latency = if total > 0 {
                stats_latency_sum / total as f32
            } else {
                0.0
            };

            eprintln!(
                "Frame {}: {:.1} FPS | Pipeline: {} real + {} Kalman | Display: {:.1}% extrapolated | Avg Latency: {:.0}ms",
                frame_id, avg_fps, stats_processed, stats_extrapolated, display_extrapolation_rate, avg_latency
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
    let total_displayed = stats_displayed_fresh + stats_displayed_extrapolated;
    let display_extrapolation_rate = if total_displayed > 0 {
        stats_displayed_extrapolated as f32 / total_displayed as f32 * 100.0
    } else {
        0.0
    };
    let extrapolation_rate = if total > 0 {
        stats_extrapolated as f32 / total as f32 * 100.0
    } else {
        0.0
    };
    let avg_latency = if total > 0 {
        stats_latency_sum / total as f32
    } else {
        0.0
    };

    eprintln!("\n📊 Final Statistics:");
    eprintln!("  • Total frames displayed: {}", frame_id);
    eprintln!("  • Duration: {:.1}s", total_time);
    eprintln!("  • Average FPS: {:.1}", avg_fps);
    eprintln!("\n  Pipeline Stats:");
    eprintln!("    - Real detections: {}", stats_processed);
    eprintln!(
        "    - Kalman predictions: {} ({:.1}%)",
        stats_extrapolated, extrapolation_rate
    );
    eprintln!("    - Average latency: {:.0}ms", avg_latency);
    eprintln!("\n  Display Stats:");
    eprintln!(
        "    - Fresh frames: {} ({:.1}%)",
        stats_displayed_fresh,
        100.0 - display_extrapolation_rate
    );
    eprintln!(
        "    - Extrapolated frames: {} ({:.1}%)",
        stats_displayed_extrapolated, display_extrapolation_rate
    );
    io::stderr().flush()?;

    // Close video writer channel and wait for thread to finish
    drop(frame_tx);
    if let Some(handle) = writer_handle {
        if let Err(e) = handle.join() {
            eprintln!("⚠️  Writer thread error: {:?}", e);
        }
    }

    eprintln!("\n✅ Output video saved: {}", output_path);
    io::stderr().flush()?;

    highgui::destroy_all_windows()?;
    Ok(())
}
