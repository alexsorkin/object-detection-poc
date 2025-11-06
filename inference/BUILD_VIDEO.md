## Building with OpenCV Video Support

### Prerequisites

**Install OpenCV and LLVM via Homebrew**:
```bash
brew install opencv llvm
```

Verify installations:
```bash
brew --prefix opencv  # Should show /usr/local/opt/opencv
brew --prefix llvm    # Should show /usr/local/opt/llvm
```

### Building

OpenCV is now a standard dependency. Simply build with:

```bash
# Build video example
cargo build --release --features metal --example detect_video

# Run video example
cargo run --release --features metal --example detect_video -- test_data/video.mp4

# Or use webcam (camera ID 0)
cargo run --release --features metal --example detect_video -- 0
```

### How It Works

The `.cargo/config.toml` file sets build-time environment variables:
```toml
[env]
DYLD_LIBRARY_PATH = { value = "/usr/local/opt/llvm/lib", force = true }
LIBCLANG_PATH = { value = "/usr/local/opt/llvm/lib", force = true }
LLVM_CONFIG_PATH = { value = "/usr/local/opt/llvm/bin/llvm-config", force = true }
OpenCV_DIR = { value = "/usr/local/opt/opencv", force = true }
```

The `force = true` ensures these variables are set even during build script execution, which is needed for opencv-rust binding generation.

### Troubleshooting

**Error: "Library not loaded: @rpath/libclang.dylib"**

This should not happen with the configured `.cargo/config.toml`, but if it does:
```bash
# Verify the config is correct
cat .cargo/config.toml

# Manually export variables and try again
export DYLD_LIBRARY_PATH="/usr/local/opt/llvm/lib:$DYLD_LIBRARY_PATH"
export LIBCLANG_PATH="/usr/local/opt/llvm/lib"
cargo build --release --features "metal,video" --example detect_video
```

**Error: "opencv not found"**
```bash
# Install opencv
brew install opencv

# Verify installation
pkg-config --modversion opencv4
```

**Error: "clang-sys build failed"**
```bash
# Install/reinstall LLVM
brew reinstall llvm

# Add to your shell profile (~/.zshrc or ~/.bash_profile):
export PATH="/usr/local/opt/llvm/bin:$PATH"
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
```

### Features

- `metal`: Enable CoreML/Metal GPU acceleration (default for macOS)
- `cuda`: Enable CUDA GPU acceleration (NVIDIA GPUs)
- `tensorrt`: Enable TensorRT acceleration (NVIDIA GPUs, optimized)

**Note**: OpenCV video support is always enabled. The above features only control GPU acceleration backend.

### Examples Without Video Support

If you don't need video processing, you can still use non-video examples:

```bash
# Test Kalman tracker (no video file required)
cargo run --release --features metal --example test_kalman

# Single image detection
cargo run --release --features metal --example detect_pipeline -- test_data/image.jpg
```

**Note**: Even though opencv is a dependency, it only adds ~2 minutes to first build (cached afterwards).
