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
use military_target_detector::kalman_pool::{KalmanJob, KalmanPool};
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
use std::sync::{Arc, Mutex};
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

    // Create dedicated Kalman worker pool (20 workers for fast parallel extrapolation)
    let kalman_tracker = Arc::new(Mutex::new(MultiObjectTracker::new(KalmanConfig::default())));
    let kalman_pool = Arc::new(KalmanPool::new(20, Arc::clone(&kalman_tracker)));

    eprintln!("  ✓ Pipeline ready");
    eprintln!("\n💡 Configuration:");
    eprintln!("  • Detector: RT-DETR");
    eprintln!("  • Workers: {}", num_workers);
    eprintln!("  • Batch size: {}", batch_size);
    eprintln!("  • Confidence: {:.0}%", confidence_threshold * 100.0);
    eprintln!("  • Classes: {:?}", allowed_classes);
    eprintln!("  • Kalman workers: 20 (dedicated pool)");
    eprintln!("  • Output FPS: {:.1} (matching input)", fps);
    eprintln!("  • Max latency for extrapolation: 500ms");

    eprintln!("\n🎬 Starting video processing...");
    eprintln!("  Press 'q' to quit, 'p' to pause\n");
    io::stderr().flush()?;

    // Create video writer for output
    let output_path = "output_video.mp4";
    // Use H.264 codec (avc1) for better compatibility
    let fourcc = VideoWriter::fourcc('a', 'v', 'c', '1')?;
    let mut writer = VideoWriter::new(
        output_path,
        fourcc,
        fps,
        opencv::core::Size::new(frame_width, frame_height),
        true,
    )?;

    if !writer.is_opened()? {
        eprintln!("❌ Failed to open video writer");
        io::stderr().flush()?;
        return Ok(());
    }

    eprintln!("  ✓ Writing output to: {}\n", output_path);
    io::stderr().flush()?;

    // Create async video writer thread
    let (frame_tx, frame_rx) = sync_channel::<Mat>(30); // Buffer 30 frames
    let writer_handle = thread::spawn(move || -> Result<(), String> {
        for mat in frame_rx {
            writer.write(&mat).map_err(|e| e.to_string())?;
        }
        writer.release().map_err(|e| e.to_string())?;
        Ok(())
    });

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

    // Track previous positions to calculate velocities for smooth interpolation
    let mut prev_positions: std::collections::HashMap<String, (f32, f32, Instant)> =
        std::collections::HashMap::new();

    eprintln!(
        "📝 Adaptive submission: max {} frames in flight, Kalman pool for fast predictions\n",
        max_pending
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

        // Consume all available results from both pipelines

        // 1. Get results from detection pipeline
        while let Some(result) = rt_pipeline.try_get_result() {
            if result.is_extrapolated {
                stats_extrapolated += 1;
            } else {
                stats_processed += 1;
            }
            stats_latency_sum += result.latency_ms;

            // Update Kalman tracker with real detections
            if !result.is_extrapolated {
                let mut tracker = kalman_tracker.lock().unwrap();
                let dt = frame_start.duration_since(last_kalman_update).as_secs_f32();
                tracker.update(&result.detections, dt);
                last_kalman_update = frame_start;
            }

            // Update position tracking for velocity calculation
            for det in &result.detections {
                let key = format!("{}_{}", det.class_id, det.tile_idx);
                prev_positions.insert(key, (det.x, det.y, result.timestamp));
            }

            latest_result = Some(result.clone());
            latest_result_time = result.timestamp;
            pending_count = pending_count.saturating_sub(1);
        }

        // 2. Get results from Kalman pool
        while let Some(kalman_result) = kalman_pool.try_get_result() {
            stats_extrapolated += 1;

            // Convert KalmanResult to FrameResult format
            let result = FrameResult {
                frame_id: kalman_result.frame_id,
                timestamp: kalman_result.timestamp,
                detections: kalman_result.detections,
                is_extrapolated: true,
                processing_time_ms: kalman_result.latency_ms,
                latency_ms: frame_start
                    .duration_since(kalman_result.timestamp)
                    .as_millis() as f32,
            };

            latest_result = Some(result.clone());
            latest_result_time = result.timestamp;
        }

        // Decide whether to submit for real detection or Kalman prediction
        let should_detect = frame_id % 10 == 0; // Real detection every 10 frames

        if should_detect && pending_count < max_pending {
            // Submit to detection pipeline
            let frame = Frame {
                frame_id,
                image: rgb_image.clone(),
                timestamp: frame_start,
            };
            if rt_pipeline.try_submit_frame(frame) {
                pending_count += 1;
            }
        } else if kalman_pool.has_capacity() {
            // Submit to Kalman pool for fast prediction
            let dt = frame_start.duration_since(last_kalman_update).as_secs_f32();
            let job = KalmanJob {
                frame_id,
                timestamp: frame_start,
                dt,
            };
            kalman_pool.try_submit(job);
        }

        // Annotate current frame with latest available detections
        let mut annotated = rgb_image.clone();

        if let Some(ref result) = latest_result {
            // Calculate time elapsed since this result was generated
            let dt = frame_start.duration_since(latest_result_time).as_secs_f32();

            // Track display stats: fresh result vs extrapolated
            if dt < 0.05 {
                // Fresh result (< 50ms old)
                stats_displayed_fresh += 1;
            } else {
                // Old result being extrapolated
                stats_displayed_extrapolated += 1;
            }

            // Calculate scale factors (detections are in resized image coordinates)
            // Pipeline resizes so SHORTER dimension = 640, preserving aspect ratio
            let tile_size = 640.0;

            // Calculate the scale factor based on the shorter dimension (matches pipeline logic)
            let scale = if orig_width < orig_height {
                tile_size / orig_width as f32
            } else {
                tile_size / orig_height as f32
            };

            // Calculate the resized dimensions
            let resized_width = (orig_width as f32 * scale) as u32;
            let resized_height = (orig_height as f32 * scale) as u32;

            // Scale factors to convert from resized coords back to original
            let scale_x = orig_width as f32 / resized_width as f32;
            let scale_y = orig_height as f32 / resized_height as f32;

            // Draw detections with coordinate scaling and temporal extrapolation
            for det in &result.detections {
                // Use Kalman velocity if available, otherwise calculate from position history
                let (extrapolated_x, extrapolated_y) = if let (Some(vx), Some(vy)) =
                    (det.vx, det.vy)
                {
                    // Use Kalman filter velocity to extrapolate forward by dt
                    if dt > 0.0 {
                        (det.x + vx * dt, det.y + vy * dt)
                    } else {
                        (det.x, det.y)
                    }
                } else {
                    // Fallback: calculate velocity from position history
                    let key = format!("{}_{}", det.class_id, det.tile_idx);
                    if let Some((prev_x, prev_y, prev_time)) = prev_positions.get(&key) {
                        if dt > 0.0 && result.timestamp > *prev_time {
                            let dt_prev = result.timestamp.duration_since(*prev_time).as_secs_f32();
                            if dt_prev > 0.0 {
                                let vx = (det.x - prev_x) / dt_prev;
                                let vy = (det.y - prev_y) / dt_prev;
                                (det.x + vx * dt, det.y + vy * dt)
                            } else {
                                (det.x, det.y)
                            }
                        } else {
                            (det.x, det.y)
                        }
                    } else {
                        (det.x, det.y)
                    }
                };

                // Use consistent color per class, with brightness indicating real vs extrapolated
                let base_color = generate_class_color(det.class_id);
                let color = if result.is_extrapolated || dt > 0.05 {
                    // Dim the color for extrapolations or old detections (multiply by 0.7)
                    Rgb([
                        (base_color.0[0] as f32 * 0.7) as u8,
                        (base_color.0[1] as f32 * 0.7) as u8,
                        (base_color.0[2] as f32 * 0.7) as u8,
                    ])
                } else {
                    base_color
                };

                // Scale detection coordinates back to original image size
                let scaled_x = (extrapolated_x * scale_x) as i32;
                let scaled_y = (extrapolated_y * scale_y) as i32;
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

                // Fix label formatting - cleaner format without confusing frame_id
                let label = if result.is_extrapolated {
                    format!("{} {:.0}% [K]", det.class_name, det.confidence * 100.0)
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
            let mode_text = if result.is_extrapolated {
                format!("EXTRAPOLATED ({} tracks)", result.detections.len())
            } else {
                format!("DETECTED ({} objects)", result.detections.len())
            };

            let stats_text = format!(
                "Frame: {} | Latency: {:.0}ms | {}",
                frame_id, result.latency_ms, mode_text
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

        // Write to output video asynchronously
        frame_tx.send(display_mat.clone()).ok();

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
    if let Err(e) = writer_handle.join() {
        eprintln!("⚠️  Writer thread error: {:?}", e);
    }

    eprintln!("\n✅ Output video saved: {}", output_path);
    io::stderr().flush()?;

    highgui::destroy_all_windows()?;
    Ok(())
}
