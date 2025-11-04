# Military Target Detector - Rust Inference

Real-time military target detection for video streams with cross-platform support.

## ⚠️ GPU Status (Important)

**Current limitation**: candle-onnx loads model weights on CPU only. For real GPU acceleration on AMD Radeon Pro 5500M:

- ✅ **CPU (Tract)**: 2,500ms - Working now
- ❌ **GPU (Candle+ONNX)**: Not supported (candle-onnx limitation)  
- 🚀 **GPU (Native Candle)**: ~100ms - Requires implementing YOLOv8 natively (1-2 days)

See `GPU_ACCELERATION_ROADMAP.md` for details and implementation plan.

## Features

- **Pure Rust**: No Python dependencies, safe for production
- **CPU Inference**: Tract ONNX Runtime (works everywhere)
- **Unity Ready**: C FFI bridge for Unity/VR integration
- **ONNX Models**: YOLOv8 nano (fast) or medium (accurate)

## Quick Start

### CPU Inference (Production Ready)
```bash
cargo build --release
cargo run --release --example detect -- ../models/military_target_detector.onnx test_data/test_tank.jpg
```

## Performance

| Platform | Hardware | Backend | Inference Time | FPS |
|----------|----------|---------|----------------|-----|
| Mac | Intel i9-9980HK | CPU (Tract) | 2,500ms | 0.4 |
| Mac | Intel i9-9980HK | CPU + Accelerate | TBD | TBD |
| Mac | AMD Radeon Pro 5500M | Native Candle (TODO) | ~100ms | ~10 |
| Python | AMD Radeon Pro 5500M | PyTorch MPS | 82.6ms | 12.1 |

## Unity Integration

Build shared library:
```bash
cargo build --release
```

This creates `libmilitary_target_detector.dylib` (macOS) or `.so` (Linux).

Use from Unity via C FFI:
```csharp
[DllImport("military_target_detector")]
private static extern IntPtr detector_new(string model_path);

[DllImport("military_target_detector")]
private static extern int detector_detect(IntPtr detector, byte[] image_data, int width, int height);
```

See `src/ffi.rs` for complete C API.

## Models

- **YOLOv8n** (nano): 12MB, 533ms CPU, fast but needs training
- **YOLOv8m** (medium): 99MB, 2,500ms CPU, pre-trained and accurate

Located in `../models/`:
- `military_target_detector.onnx` (medium, trained)
- `military_target_detector_nano.onnx` (nano, needs training)

## Architecture

- `src/detector_tract.rs` - CPU inference using Tract ONNX (✅ working)
- `src/detector_candle.rs` - GPU inference attempt (❌ limited by candle-onnx)
- `src/ffi.rs` - C FFI bridge for Unity
- `examples/detect.rs` - Single image detection
- `examples/benchmark.rs` - Performance testing

## GPU Roadmap

For real GPU acceleration on AMD Radeon Pro 5500M:

1. **Option 1 (Recommended)**: Implement native YOLOv8 in Candle
   - Pure Rust, full GPU support
   - Expected: ~100ms inference
   - Effort: 1-2 days

2. **Option 2**: Use Python bridge (not preferred)
   - Works now (82.6ms)
   - Not pure Rust

See `GPU_ACCELERATION_ROADMAP.md` for implementation details.

## Development Status

- ✅ CPU inference working (Tract)
- ✅ C FFI bridge for Unity
- ✅ YOLOv8 models exported
- ⏳ GPU acceleration (needs native Candle implementation)
- 📋 Training pipeline (Python, see ../training/)

## Building from Source

Requirements:
- Rust 1.70+
- macOS: Xcode Command Line Tools (for Metal support)

```bash
git clone <repo>
cd inference
cargo test
cargo build --release
```

## License

MIT OR Apache-2.0
