# Kalman Filter Tracking for Real-Time Video Detection

## Overview

The Kalman filter tracker enables **real-time 30 FPS output** even when the detection pipeline only processes at 5-10 FPS. It uses motion prediction to extrapolate object positions during dropped frames.

## Architecture

```
Camera (30 FPS) → Frame Buffer → Detection Pipeline (5-10 FPS) → Kalman Tracker
                                          ↓                              ↓
                                   Real Detections              Extrapolated Detections
                                          ↓                              ↓
                                   Output Stream (30 FPS combined)
```

## Components

### 1. KalmanFilter (`src/kalman_tracker.rs`)

**State Vector**: `[x, y, w, h, vx, vy]`
- `x, y`: Center position (pixels)
- `w, h`: Bounding box size (pixels)
- `vx, vy`: Velocity (pixels/second)

**Motion Model**: Constant velocity
```rust
x' = x + vx * dt
y' = y + vy * dt
w' = w (size constant)
h' = h
```

**Key Methods**:
- `predict(dt)`: Propagate state forward by `dt` seconds
- `update(detection)`: Correct prediction with new measurement

### 2. MultiObjectTracker

**Features**:
- **Data Association**: Hungarian algorithm matches detections to tracks using IoU
- **Track Management**: Auto-creates new tracks, removes stale tracks (>500ms)
- **Multi-Class Support**: Tracks different object classes independently

**Configuration**:
```rust
KalmanConfig {
    process_noise_pos: 0.5,    // Position uncertainty (pixels)
    process_noise_vel: 2.0,    // Velocity uncertainty (pixels/sec)
    measurement_noise: 2.0,    // Detector bbox jitter (pixels)
    initial_covariance: 10.0,  // Initial uncertainty
    max_age_ms: 500,           // Drop tracks after 500ms
    iou_threshold: 0.3,        // Match threshold (30% IoU)
}
```

### 3. RealtimePipeline (`src/realtime_pipeline.rs`)

**Features**:
- Async frame processing with crossbeam channels
- Automatic switching between detection and extrapolation
- Latency monitoring (switch to extrapolation if >500ms behind)

**Usage**:
```rust
let rt_pipeline = RealtimePipeline::new(pipeline, rt_config);

// Submit frames
rt_pipeline.submit_frame(frame)?;

// Get results (real or extrapolated)
if let Some(result) = rt_pipeline.get_result() {
    if result.is_extrapolated {
        println!("Kalman prediction");
    } else {
} else {
    println!("Real detection");
}
```

## Performance Results

### Test Scenario (test_kalman.rs)
- **Camera FPS**: 30
- **Processing FPS**: 5 (detection every 6 frames)
- **Motion**: Linear movement at 100 pixels/second

### Results
```
Frame   0 [DETECTED]: x=100.0
Frame   1-5 [EXTRAPOLATED]: error < 17px
Frame   6 [DETECTED]: x=119.8
Frame   7-11 [EXTRAPOLATED]: error < 15px
Frame  12 [DETECTED]: x=139.6
...
Frame  54 [DETECTED]: x=278.2
Frame  55-59 [EXTRAPOLATED]: error < 0.7px  ✓ Converged!
```

**Key Observations**:
1. **Initial frames**: High error (3-17px) - Kalman learning velocity
2. **After 3-4 detections**: Error < 5px - velocity learned
3. **Steady state**: Error < 1px - excellent tracking
4. **Confidence decay**: 0.9 → 0.7 over 5 frames

## Integration with Detection Pipeline

### Example: Video Processing

```rust
use military_target_detector::{
    detector_pool::DetectorPool,
    pipeline::{DetectionPipeline, PipelineConfig},
    realtime_pipeline::{RealtimePipeline, RealtimePipelineConfig},
};

// Create detection pipeline (existing code)
let detector_pool = Arc::new(DetectorPool::new(...)?);
let pipeline = Arc::new(DetectionPipeline::new(detector_pool, config));

// Wrap with real-time tracker
let rt_config = RealtimePipelineConfig {
    max_latency_ms: 500,
    kalman_config: Default::default(),
    buffer_size: 30,
};
let rt_pipeline = RealtimePipeline::new(pipeline, rt_config);

// Process video
for frame_id in 0.. {
    let frame = capture_frame()?;
    if !rt_pipeline.submit_frame(Frame { frame_id, image: frame, timestamp: Instant::now() }) {
        // Frame was dropped due to backpressure, advance tracker
        rt_pipeline.advance_tracks();
    }
    
    if let Some(result) = rt_pipeline.get_result() {
        visualize(result.detections, result.is_extrapolated);
    }
}
```

## Visual Indicators

When rendering detections:
- **GREEN boxes**: Real detections from neural network
- **YELLOW boxes**: Kalman filter extrapolations
- **Label suffix "(K)"**: Indicates extrapolated detection

## Tuning Parameters

### Process Noise
- **Low (0.1-0.5)**: Smooth motion (drones, vehicles on road)
- **High (1.0-5.0)**: Erratic motion (pedestrians, animals)

### Measurement Noise
- **Low (1.0-2.0)**: Accurate detector (high confidence)
- **High (3.0-10.0)**: Noisy detector (low confidence, occlusions)

### Max Age
- **Short (200-300ms)**: Fast-moving objects, reduce false tracks
- **Long (500-1000ms)**: Handle brief occlusions

## Advantages

1. **Maintains Output FPS**: 30 FPS display even with 5 FPS processing
2. **Smooth Motion**: Eliminates jitter from intermittent detections
3. **Low Latency**: Extrapolation adds <1ms overhead
4. **Occlusion Handling**: Tracks persist during brief occlusions
5. **No Training Required**: Algorithmic, no ML models needed

## Limitations

1. **Linear Motion Only**: Constant velocity model (no acceleration)
2. **Prediction Degrades**: Error grows without measurements
3. **New Objects**: Need 1-2 detections before good tracking
4. **Complex Motion**: Poor for zigzag/circular motion

## Future Enhancements

1. **Extended Kalman Filter**: Handle acceleration (curved paths)
2. **Deep SORT**: Add appearance features for re-identification
3. **Multi-Camera Fusion**: Track objects across camera views
4. **IMU Integration**: Use gyro/accelerometer for camera motion compensation

## Testing

```bash
# Run Kalman tracker test
cargo run --release --features metal --example test_kalman

# Expected output: 60 frames with extrapolation error < 1px at steady state
```

## References

- Kalman Filter: https://en.wikipedia.org/wiki/Kalman_filter
- Hungarian Algorithm: https://en.wikipedia.org/wiki/Hungarian_algorithm
- SORT Tracker: https://arxiv.org/abs/1602.00763
- Deep SORT: https://arxiv.org/abs/1703.07402
