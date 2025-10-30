# Military Target Detection Model

A real-time computer vision system for detecting military targets in live video streams, optimized for deployment across various hardware platforms including VR headsets and Unity applications.

## 🎯 Quick Start - Tank Detection

Train a tank detection model in minutes:

```bash
# Quick test run (5-10 minutes on Mac M1/M2)
./quickstart.sh

# Or run complete workflow
python workflow.py --all --epochs 50

# Test live video detection
python tests/test_tank_detection.py --mode video
```

See **[TANK_DETECTION_WORKFLOW.md](TANK_DETECTION_WORKFLOW.md)** for complete documentation.

## Target Classes
- **Person with Rifle** (armed personnel)
- **Person with Launcher** (RPG, anti-tank weapons)
- **Military Vehicle** (Russian tanks: T-62, T-64, T-72, T-80, T-90 series)

## Architecture
- **Model**: YOLOv8 for real-time object detection
- **Training**: Python with PyTorch/Ultralytics
- **Inference**: Rust library with ONNX Runtime
- **Integration**: C# bindings for Unity/VR applications

## Project Structure
```
├── training/               # Python training pipeline
│   └── train.py           # YOLOv8 training with Mac GPU
├── scripts/               # Data collection and generation
│   ├── download_tank_images.py     # Tank image collection
│   └── generate_battle_scenes.py   # Synthetic data generation
├── tests/                 # Testing suite
│   └── test_tank_detection.py     # Complete testing pipeline
├── inference/            # Rust inference library  
│   ├── src/             # Core library
│   ├── examples/        # Usage examples
│   └── bindings/        # Unity C# bindings
├── models/              # Trained models and ONNX exports
├── data/                # Training data and annotations
│   ├── raw_images/tanks/        # Downloaded tank images
│   ├── backgrounds/             # Battlefield backgrounds
│   └── synthetic_scenes/        # Generated training data
├── workflow.py          # Complete workflow orchestration
├── quickstart.sh        # Quick start script
└── TANK_DETECTION_WORKFLOW.md   # Detailed documentation
```

## Requirements
- Real-time video processing (>30 FPS)
- Cross-platform compatibility
- Low memory footprint for mobile/VR devices
- GPU acceleration support (CUDA, MPS, DirectML)
- Unity engine integration

## Mac GPU Support 🍎
✅ **Apple Silicon Macs (M1/M2/M3/M4)** - Full GPU acceleration via Metal Performance Shaders
- **10.5x faster training** than CPU (tested on Mac M1/M2)
- **89.5 FPS** training speed
- 50 epochs in ~15-20 minutes
- Unified memory architecture
- See `MAC_GPU_SETUP.md` for detailed setup instructions
- Test your system: `python test_mac_gpu.py`

## Performance Metrics

### Training (Mac M1/M2 with MPS)
- **Speed**: 89.5 FPS
- **Speedup vs CPU**: 10.5x
- **50 epochs**: ~15-20 minutes
- **Model size**: ~6 MB

### Inference
- **Latency**: 15-30 ms per frame
- **Real-time FPS**: 30-60 FPS
- **Memory usage**: <2GB RAM
- **Expected mAP@0.5**: >0.85

## Complete Workflows

### Tank Detection Workflow
```bash
# 1. Generate synthetic training data
python scripts/generate_battle_scenes.py

# 2. Train model with Mac GPU
python training/train.py --epochs 50 --device mps

# 3. Test on synthetic scenes
python tests/test_tank_detection.py --mode dataset

# 4. Test on live video
python tests/test_tank_detection.py --mode video

# 5. Export to ONNX for Rust integration
python workflow.py --export
```

See **[TANK_DETECTION_WORKFLOW.md](TANK_DETECTION_WORKFLOW.md)** for detailed documentation.

## Key Features

### 🎨 Synthetic Data Generation
- Realistic battlefield scenes with tanks
- 5 background types (field, urban, forest, desert, mountain)
- Automatic YOLO format annotations
- Random augmentation (rotation, scaling, lighting)

### 🚀 High-Performance Training
- YOLOv8 real-time detection
- Mac GPU acceleration (MPS)
- Automatic device detection
- Real-time training metrics

### 🧪 Comprehensive Testing
- Batch evaluation with metrics
- Single image testing
- Live video stream inference
- Precision, Recall, F1 scores

### 🦀 Rust Inference Library
- Cross-platform ONNX Runtime
- C API for Unity/VR integration
- C# bindings for Unity
- Hardware-agnostic design

## Performance Targets
- Inference: <33ms per frame (30+ FPS)
- Memory usage: <2GB RAM
- Model size: <100MB for deployment

## Quick Setup & Dependency Management

### Option 1: Automated Setup (Recommended)
```bash
# Full setup with latest dependencies
./setup.sh

# Or upgrade existing installation
./upgrade_dependencies.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Choose appropriate requirements file:
pip install -r requirements.txt              # Latest versions (recommended)
pip install -r requirements-conservative.txt # Compatible with older systems
```

### Requirements Files
- **`requirements.txt`** - Latest stable versions with newest features
- **`requirements-conservative.txt`** - Conservative versions for compatibility
- **`requirements-lock.txt`** - Exact versions (generated after setup)