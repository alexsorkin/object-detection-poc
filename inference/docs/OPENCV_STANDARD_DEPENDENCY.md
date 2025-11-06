# OpenCV as Standard Dependency - Summary

## ✅ Changes Made

### 1. **Removed Feature Gate**
OpenCV is now a **standard dependency**, not optional:

```toml
# Before (optional)
opencv = { version = "0.93", ..., optional = true }
[features]
video = ["opencv"]

# After (always enabled)
opencv = { version = "0.93", ... }
[features]
# video feature removed
```

### 2. **Simplified Build Commands**

All examples now use the same feature flags:

```bash
# Video processing
cargo run --release --features metal --example detect_video -- video.mp4

# Kalman test
cargo run --release --features metal --example test_kalman

# Single image
cargo run --release --features metal --example detect_pipeline -- image.jpg
```

**No more `--features "metal,video"`** - just `--features metal`!

### 3. **Updated Documentation**

- ✅ `BUILD_VIDEO.md` - Removed video feature flag references
- ✅ `README.md` - Simplified commands
- ✅ `OPENCV_SETUP.md` - Updated usage examples

## 🎯 Rationale

### Why Make OpenCV Standard?

1. **Simpler Mental Model**
   - Video support is a core feature, not optional
   - Kalman tracking benefits from video processing utilities
   - Reduces confusion about which features to enable

2. **Better Developer Experience**
   - One less thing to remember
   - Consistent build commands across examples
   - OpenCV build is cached after first compile (~2 min)

3. **Preparation for Production**
   - Real-world deployments need video support
   - Optional dependencies create deployment complexity
   - Better to handle opencv setup once upfront

## 📦 Build Impact

### First Build (Cold)
```
Total time: ~3-4 minutes
- opencv binding generation: ~2 min
- Other dependencies: ~1-2 min
```

### Incremental Builds (After Changes)
```
Total time: ~30-60 seconds
- opencv cached, no rebuild needed
```

### File Size
```
Binary size increase: ~5-10 MB
- opencv bindings: minimal overhead
- Shared system opencv libraries used at runtime
```

## 🚀 Migration Guide

### For Existing Code

**Before:**
```bash
cargo build --release --features "metal,video"
```

**After:**
```bash
cargo build --release --features metal
```

### For CI/CD

Ensure opencv + llvm are installed:
```bash
# macOS
brew install opencv llvm

# Linux
apt-get install libopencv-dev llvm-dev libclang-dev

# Windows
vcpkg install opencv llvm
```

## ✅ Benefits

1. **Consistency**: Same feature flags for all examples
2. **Simplicity**: No need to remember video feature
3. **Completeness**: Full functionality always available
4. **Performance**: No runtime overhead (unused code is optimized out)

## 📝 Current Feature Flags

```toml
[features]
default = []
metal = []      # CoreML/Metal GPU (macOS)
cuda = []       # CUDA GPU (NVIDIA)
tensorrt = []   # TensorRT optimized (NVIDIA)
```

OpenCV is always included, GPU backend is selected by feature flag.
