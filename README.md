# Military Target Detection Model

High-performance computer vision system for detecting military targets using RT-DETR and Rust inference.

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd scripts
./setup.sh
source ../venv/bin/activate
```

### 2. Export Models
```bash
# Export mini model (fast, 43MB, ~180ms)
python export_mini.py

# Export full model (accurate, 99MB, ~490ms)
python export_full.py
```

### 3. Run Detection (Rust)
```bash
cd ../inference

# Fast detection (recommended)
cargo run --release --features metal --example detect_mini

# Accurate detection
cargo run --release --features metal --example detect_full

# Benchmark both models
cargo run --release --features metal --example benchmark_all
```

## 📦 Models

| Model | Size | Speed | FPS | Use Case |
|-------|------|-------|-----|----------|
| **RT-DETR r18** ⭐ | ~70MB | ~45ms | 22 | Real-time, recommended |
| **RT-DETR r50** | ~150MB | ~90ms | 11 | Maximum accuracy |

*Benchmarked on AMD Radeon Pro 5500M with CoreML/Metal*

## Architecture

- **Training**: Python with PyTorch/Ultralytics (RT-DETR)
- **Inference**: Rust with ONNX Runtime + CoreML/Metal GPU
- **Integration**: C/C# bindings for Unity applications
- **Platform**: macOS (Metal), iOS, visionOS (Apple Vision Pro)

## Project Structure
```
model/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── models/                # Deployed ONNX models
│   ├── military_target_detector_mini.onnx  # 43MB, fast
│   └── military_target_detector.onnx       # 99MB, accurate
├── inference/             # 🦀 Rust inference library
│   ├── src/              # Core detection engine
│   ├── examples/         # Usage examples
│   │   ├── detect_mini.rs        # Fast detection
│   │   ├── detect_full.rs        # Accurate detection
│   │   └── benchmark_all.rs      # Performance testing
│   └── Cargo.toml        # Rust dependencies
├── scripts/               # Utility scripts
│   ├── generate_battle_scenes.py
│   ├── create_test_image.py
│   └── setup.sh          # Environment setup
├── training/              # Training pipeline
│   ├── train.py          # RT-DETR training
│   └── export.py         # Model export
├── data/                  # Training datasets
└── bindings/              # C/Unity bindings
```

## GPU Acceleration 🚀

### Apple Silicon (M1/M2/M3/M4) - CoreML/Metal
- **Automatic GPU selection** - Uses AMD Radeon Pro or integrated GPU
- **Fast inference** - 180ms (Mini), 496ms (Full)
- **Build with GPU**: `cargo build --release --features metal`
- **Platforms supported**: macOS, iOS, visionOS (Apple Vision Pro)

### Performance Metrics
- **RT-DETR r18**: ~45ms avg, 22 FPS (recommended)
- **RT-DETR r50**: ~90ms avg, 11 FPS (accurate)
- **GPU**: AMD Radeon Pro 5500M (8GB) / Apple Silicon
- **API**: ONNX Runtime with CoreML backend

## Key Features

### 🦀 Rust Inference Library
- Fast ONNX Runtime integration
- CoreML/Metal GPU acceleration  
- Thread-safe detection API
- C/Unity bindings included
- Cross-platform support

### 📦 Optimized Models
- Multiple RT-DETR variants (r18/r34/r50/r101)
- Transformer-based detection (no NMS required)
- ONNX format for portability
- Efficient memory usage

### 🎯 Real-time Detection
- Video processing support (OpenCV)
- Batch detection capability
- Configurable confidence thresholds
- Non-maximum suppression

## Examples

### Rust Detection
```rust
use military_target_detector::{DetectorConfig, MilitaryTargetDetector, ImageData};

let config = DetectorConfig {
    model_path: "../models/military_target_detector_mini.onnx".to_string(),
    ..Default::default()
};

let mut detector = MilitaryTargetDetector::new(config)?;
let image = ImageData::from_file("test.jpg")?;
let detections = detector.detect(&image)?;
```

### Command Line
```bash
# Fast detection
cargo run --release --features metal --example detect_mini image.jpg

# Video processing (requires opencv feature)
cargo run --release --features metal,opencv --example video_stream

# Performance benchmark
cargo run --release --features metal --example benchmark_all
```

## Unity Integration

1. **Copy bindings**: `bindings/MilitaryTargetDetector.cs` → Unity project
2. **Copy library**: Compiled `.dylib`/`.dll` → `Assets/Plugins/`
3. **Copy model**: ONNX model → `Assets/StreamingAssets/Models/`
4. **Use in Unity**:
```csharp
var detector = new MilitaryTargetDetector(config);
var detections = detector.Detect(imageBytes, width, height);
```

## Development

### Build
```bash
cd inference
cargo build --release --features metal
```

### Test
```bash
cargo test --features metal
cargo run --example detect_mini
```

### Benchmark
```bash
cargo run --release --features metal --example benchmark_all
```