# Kalman Filter Integration with Persistent Track IDs

## Overview

Successfully integrated `MultiObjectTracker` with persistent track ID support into the video detection pipeline. This provides smooth object tracking and temporal extrapolation for RT-DETR detections on CPU.

## Architecture

### Simple Synchronous Design

```
┌─────────────────────────────────────────┐
│         Video Frame Loop (24 FPS)       │
└─────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │ Frame #40, #80, #120, ...?        │
    │ (Every 40th frame)                │
    └───────────────────────────────────┘
            YES │         NO │
                │            │
                ▼            ▼
    ┌──────────────────┐   ┌────────────────────┐
    │ Submit to RT-DETR│   │ Skip (no detection)│
    │ Pipeline         │   └────────────────────┘
    └──────────────────┘
                │
                ▼
    ┌──────────────────────────────────┐
    │ Detection Ready?                 │
    │ (Async, ~1.5-2.5s latency)       │
    └──────────────────────────────────┘
                │
                ▼
    ┌──────────────────────────────────┐
    │ tracker.update(detections, dt)   │
    │ (Synchronous - Update Kalman)    │
    └──────────────────────────────────┘
                │
                ▼
    ┌──────────────────────────────────┐
    │ predictions = tracker.get_       │
    │              predictions()       │
    │ (Synchronous - Read state)       │
    └──────────────────────────────────┘
                │
                ▼
    ┌──────────────────────────────────┐
    │ Display with Track ID Colors     │
    └──────────────────────────────────┘
```

### Key Principles

1. **Simple**: Single-threaded `MultiObjectTracker`, no complex async pools
2. **Synchronous Updates**: Update Kalman when detection arrives (no race conditions)
3. **Synchronous Reads**: Get predictions directly from tracker state
4. **Persistent IDs**: Each object gets unique track_id, maintained across frames
5. **Clean Separation**: Detection submission and Kalman tracking are independent

## Implementation Details

### 1. Track ID Infrastructure

**TileDetection Struct** (`src/pipeline.rs`):
```rust
pub struct TileDetection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: String,
    pub tile_idx: usize,
    pub vx: Option<f32>,      // Kalman velocity X
    pub vy: Option<f32>,      // Kalman velocity Y
    pub track_id: Option<u32>, // Persistent object ID
}
```

**Raw Detections** (line 484):
```rust
TileDetection {
    // ... other fields ...
    vx: None,
    vy: None,
    track_id: None,  // No ID yet, assigned by tracker
}
```

### 2. MultiObjectTracker Integration

**TrackedObject** (`src/kalman_tracker.rs`):
```rust
pub struct TrackedObject {
    pub track_id: u32,           // Unique persistent ID
    pub kalman: KalmanFilter,    // 6D state [x,y,w,h,vx,vy]
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub last_update: Instant,
}

impl TrackedObject {
    pub fn get_detection(&self) -> TileDetection {
        let mut det = self.kalman.get_detection(
            self.class_id, 
            &self.class_name, 
            self.confidence
        );
        det.track_id = Some(self.track_id);  // Assign persistent ID
        det
    }
}
```

**Track Assignment** (Hungarian algorithm with IoU matching):
```rust
pub fn update(&mut self, detections: &[TileDetection], dt: f32) {
    // 1. Predict all existing tracks forward
    // 2. Match detections to tracks using Hungarian + IoU
    // 3. Update matched tracks
    // 4. Create new tracks for unmatched detections
    // 5. Remove stale tracks (age > max_age_ms)
}
```

### 3. Video Loop Integration

**Main Loop** (`examples/detect_video.rs`):
```rust
// Create single shared tracker
let mut kalman_tracker = MultiObjectTracker::new(KalmanConfig::default());

loop {
    // 1. Get results from detection pipeline
    while let Some(result) = rt_pipeline.try_get_result() {
        if !result.is_extrapolated {
            // Update Kalman with real detections (synchronous)
            let dt = frame_start.duration_since(last_kalman_update).as_secs_f32();
            kalman_tracker.update(&result.detections, dt);
            last_kalman_update = frame_start;
        }
    }

    // 2. Submit detection every 40th frame
    if frame_id % 40 == 0 && pending_count < max_pending {
        rt_pipeline.try_submit_frame(frame);
    }

    // 3. Get Kalman predictions for display (synchronous)
    let kalman_predictions = kalman_tracker.get_predictions();

    // 4. Display with track ID colors
    for det in &kalman_predictions {
        let color = if let Some(track_id) = det.track_id {
            // Hue-based coloring: consistent color per track
            let hue = (track_id as f32 * 137.5) % 360.0;
            hue_to_rgb(hue)
        } else {
            generate_class_color(det.class_id)
        };
        
        draw_rect(/* ... with track ID color ... */);
        
        let label = format!("#{} {} {:.0}%", 
                           track_id, 
                           det.class_name, 
                           det.confidence * 100.0);
    }
}
```

### 4. Track ID Visualization

**Color Scheme**:
- Each track ID gets a unique color using golden angle hue distribution
- `hue = (track_id * 137.5) % 360.0` ensures visually distinct colors
- Same object = same color across all frames

**Label Format**:
- `#<track_id> <class_name> <confidence>%`
- Example: `#1 person 87%`, `#2 car 92%`

## Performance Characteristics

### RT-DETR on CPU
- **Detection Rate**: Every 40th frame (~14 detections for 576-frame video)
- **Latency**: 1.5-2.5 seconds per detection
- **Output**: Full 24 FPS with Kalman extrapolation

### Kalman Tracker
- **State Vector**: 6D [x, y, w, h, vx, vy]
- **Prediction**: Constant velocity model
- **Matching**: Hungarian algorithm with IoU threshold (0.3)
- **Cleanup**: Tracks removed after 500ms without update
- **Track ID Counter**: Monotonically increasing (never reused)

### Results
```
Frame 288: 22.3 FPS | Pipeline: 6 real + 263 Kalman | Display: 100.0% extrapolated
```
- **Display FPS**: ~22 FPS (close to target 24 FPS)
- **Real Detections**: 6 (every 40th frame submitted)
- **Kalman Predictions**: 263 (filling all other frames)
- **Extrapolation Rate**: 100% (using Kalman for smooth display)

## Configuration

### KalmanConfig
```rust
KalmanConfig {
    process_noise_pos: 0.5,   // Position uncertainty (pixels)
    process_noise_vel: 2.0,   // Velocity uncertainty (pixels/sec)
    measurement_noise: 2.0,   // Detector jitter (pixels)
    initial_covariance: 10.0, // Initial uncertainty
    max_age_ms: 500,          // Drop tracks after 500ms
    iou_threshold: 0.3,       // Match if IoU > 30%
}
```

### Detection Strategy
```rust
// Submit every 40th frame for RT-DETR on CPU
let should_detect = frame_id % 40 == 0;

// Max 2 frames in flight
let max_pending = 2;
```

## Benefits of Simple Architecture

### ✅ What Works
1. **No Buffer Issues**: Single tracker state, no queues or channels
2. **No Race Conditions**: Synchronous updates and reads
3. **No Disappearing Objects**: Cleanup only happens in update(), not predictions
4. **Persistent Identity**: Track IDs maintained across brief occlusions
5. **Low Latency**: Direct state access, no message passing overhead
6. **Smooth Output**: Kalman fills ~97% of frames (560/576)

### ❌ What Was Removed
1. **Complex Async Pool**: 20 worker threads with message passing
2. **Result Draining**: get_latest_result() that lost critical updates
3. **Per-Frame Requests**: Prediction requests for every frame
4. **Automatic Cleanup**: Background threads that caused race conditions
5. **Large Buffers**: 30-frame queues that accumulated stale data

## Usage

```bash
# Run with default settings (50% confidence, default classes)
cargo run --release --example detect_video test_data/airport.mp4

# Custom confidence threshold
cargo run --release --example detect_video -- --confidence 35 test_data/airport.mp4

# Filter specific classes
cargo run --release --example detect_video -- --classes 0,2,5,7 test_data/airport.mp4
```

## Output

- **Video File**: `output_video.mp4` (H.264/avc1 codec)
- **Track Visualization**: Each object has consistent color by ID
- **Stats Overlay**: Frame count, active tracks, total detections
- **Console Logs**: Real-time performance metrics

## Next Steps

### Potential Improvements
1. **Track ID Persistence**: Save/load track IDs for video restart
2. **Appearance Features**: Add visual similarity to matching (not just IoU)
3. **Motion Models**: Non-linear models for turning/accelerating objects
4. **Occlusion Handling**: Better prediction during brief occlusions
5. **Multi-Camera**: Track objects across multiple camera views

### Tuning Parameters
- `frame_id % N`: Adjust N based on CPU speed (40 for RT-DETR)
- `max_age_ms`: Increase for longer occlusions (500ms default)
- `iou_threshold`: Lower for smaller objects (0.3 default)
- `process_noise_vel`: Increase for erratic motion (2.0 default)

## Lessons Learned

1. **Keep It Simple**: Synchronous > Async for this use case
2. **No Over-Engineering**: CPU RT-DETR is fundamentally slow, accept it
3. **State Management**: Single source of truth (tracker) prevents races
4. **Cleanup Location**: Only clean up during updates, never during reads
5. **ID Assignment**: Assign once, never reassign (persistent identity)

## References

- **Kalman Filter**: `src/kalman_tracker.rs`
- **Hungarian Algorithm**: `pathfinding` crate
- **IoU Matching**: Intersection over Union for bbox association
- **Track ID**: Monotonic counter, never reused
- **Golden Angle**: 137.5° for distinct hue distribution
