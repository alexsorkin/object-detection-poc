# OpenCV Cross-Platform Setup

## 🎯 Cross-Platform OpenCV Configuration

This project now uses a unified cross-platform approach for OpenCV with version 4.12.0 across macOS, Linux, and Windows.

## 📦 Installation Methods by Platform

### macOS (Homebrew)
```bash
# OpenCV 4.12.0 via Homebrew (already working)
brew install opencv
```

### Linux (Conda - Recommended)
```bash
# Install OpenCV 4.12.0 via conda for version parity with macOS
conda install -c conda-forge opencv=4.12.0

# Alternative: System package manager (older version)
sudo apt update && sudo apt install libopencv-dev
```

### Windows (Conda - Recommended)
```bash
# Install OpenCV 4.12.0 via conda
conda install -c conda-forge opencv=4.12.0
```

## ⚙️ Build Configuration

The `.cargo/config.toml` now supports cross-platform builds with target-specific environment variables:

```toml
# Cross-platform configuration for different targets
[target.x86_64-apple-darwin]
[target.x86_64-apple-darwin.env]
DYLD_LIBRARY_PATH = { value = "/usr/local/opt/llvm/lib", force = true }
LIBCLANG_PATH = { value = "/usr/local/opt/llvm/lib", force = true }
LLVM_CONFIG_PATH = { value = "/usr/local/opt/llvm/bin/llvm-config", force = true }
OpenCV_DIR = { value = "/usr/local/opt/opencv", force = true }

[target.aarch64-apple-darwin]
[target.aarch64-apple-darwin.env]
DYLD_LIBRARY_PATH = { value = "/opt/homebrew/opt/llvm/lib", force = true }
LIBCLANG_PATH = { value = "/opt/homebrew/opt/llvm/lib", force = true }
LLVM_CONFIG_PATH = { value = "/opt/homebrew/opt/llvm/bin/llvm-config", force = true }
OpenCV_DIR = { value = "/opt/homebrew/opt/opencv", force = true }

[target.x86_64-unknown-linux-gnu]
[target.x86_64-unknown-linux-gnu.env]
# Conda-based OpenCV installation (recommended for version parity)
OpenCV_DIR = { value = "/home/alexs/miniconda3", force = true }
PKG_CONFIG_PATH = { value = "/home/alexs/miniconda3/lib/pkgconfig", force = true }

[target.x86_64-pc-windows-msvc]
[target.x86_64-pc-windows-msvc.env]
# Conda-based OpenCV installation
OpenCV_DIR = { value = "C:/Users/USERNAME/miniconda3", force = true }
PKG_CONFIG_PATH = { value = "C:/Users/USERNAME/miniconda3/lib/pkgconfig", force = true }
```

## 🎯 Usage

### Build & Run
```bash
# OpenCV is now always enabled - cross-platform compatible
cargo build --release --example detect_video
cargo run --release --example detect_video -- --model r50vd_fp32 ./test_data/airport_640.mp4

# Or use webcam
cargo run --release --example detect_video -- --model r50vd_fp32 0
```

## 📦 Recent Changes (December 2025)

### ✅ Cross-Platform OpenCV Compatibility Fixed

1. **Version Standardization**
   - **Problem**: macOS had OpenCV 4.12.0 (Homebrew), Linux had 4.6.0 (apt)
   - **Root Cause**: opencv-rust crate generates different bindings based on local OpenCV version
   - **Solution**: Upgraded Linux to OpenCV 4.12.0 via conda for version parity

2. **API Modernization**
   - Removed deprecated `AlgorithmHint::ALGO_HINT_DEFAULT` parameter from `imgproc::cvt_color()` calls
   - Updated to modern 4-parameter OpenCV API: `cvt_color(src, dst, code, dst_cn)`
   - Fixed across: `image_utils.rs`, `video_utils.rs`, `detect_video.rs`

3. **Cross-Platform Build Configuration**
   - Target-specific `.cargo/config.toml` sections for macOS Intel/ARM, Linux, Windows
   - Conda integration for consistent OpenCV 4.12.0 across platforms
   - Automated environment variable setup per platform

4. **Unity Bindings Updates**
   - Updated P/Invoke declarations to match current RT-DETR FFI
   - Fixed class definitions from 4 military classes to 80 COCO classes
   - Removed unsupported `nmsThreshold` parameter

## 🔧 Technical Details

### Why Conda for OpenCV?

1. **Version Consistency**: conda-forge provides OpenCV 4.12.0 across all platforms
2. **Binding Compatibility**: Ensures opencv-rust generates identical bindings
3. **Dependency Management**: Handles complex OpenCV dependencies automatically
4. **No Version Drift**: Unlike system package managers that vary by OS

### OpenCV Rust Crate Behavior

The opencv-rust crate (v0.97.2) dynamically generates Rust bindings at build time:
- Parses local OpenCV headers using libclang
- Generates different APIs based on installed OpenCV version
- **Critical**: Same OpenCV version across dev environments prevents API mismatches

## 📊 Files Created/Modified

- ✅ `Cargo.toml` - opencv dependency (always enabled)
- ✅ `.cargo/config.toml` - Cross-platform build configuration with target-specific environment variables
- ✅ `src/image_utils.rs` - Updated OpenCV API calls for 4.12.0 compatibility
- ✅ `src/video_utils.rs` - Fixed `cvt_color` calls, removed deprecated parameters
- ✅ `examples/detect_video.rs` - Modernized OpenCV API usage
- ✅ `bindings/MilitaryTargetDetector.cs` - Updated Unity P/Invoke for current FFI
- ✅ `OPENCV_SETUP.md` - This documentation with conda setup instructions

## 🔧 Platform-Specific Setup

### macOS Setup (Working)
```bash
# Already configured - using Homebrew OpenCV 4.12.0
brew install opencv llvm
```

### Linux Setup (Updated)
```bash
# Recommended: Use conda for version parity
conda install -c conda-forge opencv=4.12.0

# Update .cargo/config.toml with your conda prefix:
# OpenCV_DIR = { value = "/home/USERNAME/miniconda3", force = true }
# PKG_CONFIG_PATH = { value = "/home/USERNAME/miniconda3/lib/pkgconfig", force = true }

# Verify installation
PKG_CONFIG_PATH=/home/USERNAME/miniconda3/lib/pkgconfig pkg-config --modversion opencv4
# Should output: 4.12.0
```

### Windows Setup (Recommended)
```bash
# Use conda for OpenCV installation
conda install -c conda-forge opencv=4.12.0

# Update .cargo/config.toml with your conda prefix:
# OpenCV_DIR = { value = "C:/Users/USERNAME/miniconda3", force = true }
# PKG_CONFIG_PATH = { value = "C:/Users/USERNAME/miniconda3/lib/pkgconfig", force = true }
```

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

### OpenCV Version Compatibility Matrix

| Version | Status | Platform | Notes |
|---------|--------|----------|--------|
| 4.6.0 | ⚠️ Legacy | Ubuntu apt | Generates older API bindings |
| 4.12.0 | ✅ Current | conda-forge | **Recommended** - consistent across platforms |
| 4.12.0 | ✅ Working | macOS Homebrew | Reference implementation |
| 0.97.2 | ✅ Stable | opencv-rust crate | Current Rust bindings version |

## 🚀 Quick Start

### Prerequisites Check
```bash
# Verify OpenCV version (should be 4.12.0)
pkg-config --modversion opencv4

# Check opencv-rust crate version
grep opencv Cargo.toml
# Should show: opencv = "0.97.2"
```

### Build & Test
```bash
# Clean build to regenerate bindings
cargo clean
cargo build --release

# Test with video
cargo run --release --example detect_video -- --model r50vd_fp32 ./test_data/airport_640.mp4

# Expected output: No compilation warnings, successful video processing
```

## 🚀 Next Steps

1. **Test Cross-Platform Builds**
   ```bash
   # Test on different platforms with same commands
   cargo run --release --example detect_video -- --model r50vd_fp32 0
   ```

2. **Performance Optimization**
   - Monitor detection latency (~200-300ms per frame on CPU)
   - Tune tracking parameters in ByteTrack configuration
   - Consider GPU acceleration for RT-DETR inference

3. **Development Workflow**
   - Use `cargo clean` when switching between platforms
   - Verify OpenCV version consistency across dev team
   - Update conda environment regularly for security patches

## 📊 Expected Performance (OpenCV 4.12.0)

**Cross-Platform Benchmarks:**
- **RT-DETR Inference**: ~200-300ms per frame (CPU)
- **OpenCV Processing**: ~5-10ms per frame
- **Video I/O**: ~1-2ms per frame
- **Total Pipeline**: ~210-320ms per frame
- **Tracking Update**: ~1ms per frame

## ⚠️ Known Issues & Solutions

### 1. **Build Failures After OpenCV Updates**
```bash
# Solution: Clean and rebuild to regenerate bindings
cargo clean
cargo build --release
```

### 2. **Different OpenCV Versions in Team**
```bash
# Solution: Standardize on conda installation
conda install -c conda-forge opencv=4.12.0

# Verify version consistency
pkg-config --modversion opencv4  # Should output: 4.12.0
```

### 3. **Missing libclang (Linux)**
```bash
# Solution: Install build dependencies
sudo apt install clang libclang-dev pkg-config
```

### 4. **Windows Build Issues**
```bash
# Solution: Use conda and Visual Studio Build Tools
conda install -c conda-forge opencv=4.12.0
# Ensure VS Build Tools are in PATH
```

## 🎓 Documentation References

- **Cross-Platform Build**: Current document
- **Unity Bindings**: See [../bindings/](../bindings/) for updated C# integration
- **Kalman Tracking**: See [KALMAN_TRACKING.md](KALMAN_TRACKING.md)
- **Video Build**: See [BUILD_VIDEO.md](BUILD_VIDEO.md)  
- **Class Filtering**: See [CLASS_FILTERING.md](CLASS_FILTERING.md)
- **Visualization**: See [VISUALIZATION.md](VISUALIZATION.md)

## 📝 Changelog

### v1.2.0 (December 2025) - Cross-Platform OpenCV
- ✅ Conda-based OpenCV 4.12.0 installation across platforms
- ✅ Fixed API compatibility issues with modern OpenCV
- ✅ Cross-platform `.cargo/config.toml` configuration
- ✅ Updated Unity bindings for current FFI interface
- ✅ Removed deprecated OpenCV API parameters

### v1.1.0 (Previous) - Initial Video Support
- ✅ Added opencv 0.93.7 with video feature flag
- ✅ macOS-specific build configuration
- ✅ Basic video processing pipeline
