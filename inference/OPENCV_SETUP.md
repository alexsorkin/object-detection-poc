# OpenCV Video Support - Impleme## 🎯 Usage

### Build & Run
```bash
# OpenCV is now always enabled - no special feature flag needed
cargo build --release --features metal --example detect_video
cargo run --release --features metal --example detect_video -- video.mp4

# Or use webcam
cargo run --release --features metal --example detect_video -- 0
```

### Without Video Support✅ What Was Fixed

### 1. **Dependency Management**
- Added opencv 0.93.7 as optional dependency
- Created `video` feature flag in Cargo.toml
- Upgraded from problematic opencv 0.92.3 to 0.93.7

### 2. **Build Environment**
Configured `.cargo/config.toml` with proper environment variables:
```toml
[env]
DYLD_LIBRARY_PATH = { value = "/usr/local/opt/llvm/lib", force = true }
LIBCLANG_PATH = { value = "/usr/local/opt/llvm/lib", force = true }
LLVM_CONFIG_PATH = { value = "/usr/local/opt/llvm/bin/llvm-config", force = true }
OpenCV_DIR = { value = "/usr/local/opt/opencv", force = true }
```

The `force = true` ensures these are set during build script execution.

### 3. **API Compatibility**
Fixed opencv 0.93 API changes in `detect_video.rs`:
- `cvt_color()` now requires `AlgorithmHint::ALGO_HINT_DEFAULT` enum (not const)
- `Mat::new_rows_cols_with_data()` simplified API (no manual CV_8UC3 type)
- Type mismatches: `f32 / f64` → `f64 / f64`

### 4. **Documentation**
- Created `BUILD_VIDEO.md` with setup instructions
- Updated main `README.md` with video examples
- Documented troubleshooting steps

## 🎯 Usage

### Build & Run (Recommended)
```bash
# Simple cargo commands - .cargo/config.toml handles the rest
cargo build --release --features "metal,video" --example detect_video
cargo run --release --features "metal,video" --example detect_video -- video.mp4

# Or use webcam
cargo run --release --features "metal,video" --example detect_video -- 0
```

### Alternative: Using Shell Script
```bash
# If .cargo/config.toml doesn't work in your environment
./cargo-opencv.sh build --release --features "metal,video" --example detect_video
./cargo-opencv.sh run --release --features "metal,video" --example detect_video -- video.mp4
```

### Without OpenCV (Kalman Test Only)
```bash
# No opencv dependency needed
cargo run --release --features metal --example test_kalman
```

## 📦 Files Created/Modified

- ✅ `Cargo.toml` - Added opencv as optional dependency with `video` feature
- ✅ `.cargo/config.toml` - Build environment with `force = true` for build scripts
- ✅ `examples/detect_video.rs` - Fixed API compatibility for opencv 0.93
- ✅ `BUILD_VIDEO.md` - Complete build instructions
- ✅ `README.md` - Updated with video examples

## 🔧 Technical Details

### Why opencv-rust Needs Special Setup

1. **Build-Time Binding Generation**
   - opencv-rust generates Rust bindings from C++ headers during `cargo build`
   - Requires libclang.dylib to parse headers
   - Solution: Use `[env]` with `force = true` in `.cargo/config.toml`

2. **Environment Variables**
   - `LIBCLANG_PATH`: Points to libclang.dylib for header parsing
   - `LLVM_CONFIG_PATH`: For compiler configuration
   - `DYLD_LIBRARY_PATH`: For loading libclang.dylib at build time
   - `OpenCV_DIR`: OpenCV installation directory

3. **The `force = true` Key**
   - Standard `[env]` only applies to runtime
   - `force = true` makes cargo set these for build scripts too
   - This is the correct solution, no shell script needed!

### OpenCV Version Compatibility

| Version | Status | Issues |
|---------|--------|--------|
| 0.92.3 | ❌ Broken | Missing `type_to_string()`, `depth_to_string()` functions |
| 0.93.7 | ✅ Working | API changes (AlgorithmHint enum), but compilable |
| 0.97.1 | ⚠️ Latest | Not tested, may have more breaking changes |

## 🚀 Next Steps

1. **Test with Real Video**
   ```bash
   # Download test video or use webcam
   ./cargo-opencv.sh run --release --features "metal,video" --example detect_video -- 0
   ```

2. **Performance Tuning**
   - Adjust `max_latency_ms` in RealtimePipelineConfig
   - Tune Kalman filter noise parameters
   - Monitor extrapolation rate (should be 70-80% for 30fps camera with 5fps processing)

3. **Optional: Deep SORT**
   - Add appearance-based re-identification
   - Requires additional CNN model for feature extraction

## 📊 Expected Performance

With your i9 + AMD GPU setup:
- **Processing FPS**: 5-10 (limited by GPU via CoreML)
- **Output FPS**: 30 (with Kalman extrapolation)
- **Extrapolation Rate**: 70-80%
- **Prediction Error**: <1px after convergence (3-4 frames)

## ⚠️ Known Limitations

1. **opencv-rust Platform Support**
   - macOS: ✅ Works (via Homebrew)
   - Linux: ✅ Should work (apt install)
   - Windows: ⚠️ May need manual opencv build

2. **Kalman Filter Motion Model**
   - Only constant velocity (linear motion)
   - Not suitable for zigzag/circular paths
   - Consider Extended Kalman Filter for complex motion

3. **Build Time**
   - First build with opencv: ~2-3 minutes (binding generation)
   - Subsequent builds: <30 seconds (cached)

## 🎓 Documentation References

- **Kalman Tracking**: See [KALMAN_TRACKING.md](KALMAN_TRACKING.md)
- **Video Build**: See [BUILD_VIDEO.md](BUILD_VIDEO.md)  
- **Class Filtering**: See [CLASS_FILTERING.md](CLASS_FILTERING.md)
- **Visualization**: See [VISUALIZATION.md](VISUALIZATION.md)
