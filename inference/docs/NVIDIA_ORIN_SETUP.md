# NVIDIA Orin NX Setup Guide

This guide explains how to deploy the military target detector on NVIDIA Orin NX hardware for maximum performance with FP16 acceleration.

## Why NVIDIA Orin NX is Faster with FP16

### Performance Benefits:
- **2x FP16 Throughput**: Orin NX has dedicated FP16 tensor cores with ~2x the operations/second vs FP32
- **50% Less Memory**: FP16 model is 50MB vs 99MB FP32, reducing memory bandwidth bottlenecks
- **Native Support**: CUDA and TensorRT fully support FP16 without type conversion issues
- **Expected Speedup**: 1.5-2.5x faster inference compared to FP32

## Model Selection for Orin NX

```rust
let detector_config = DetectorConfig {
    fp16_model_path: Some("../models/yolov8m_batch_fp16.onnx".to_string()), // Use FP16 for NVIDIA!
    fp32_model_path: Some("../models/yolov8m_batch_fp32.onnx".to_string()), // CPU fallback
    use_gpu: true,
    gpu_device_id: 0,  // Orin NX typically has one GPU
    ..Default::default()
};
```

## Build Options

### Option 1: CUDA (Recommended for Orin NX)
```bash
# Install ONNX Runtime with CUDA support (on Orin NX)
cargo build --release --features cuda --example detect_pipeline

# Run
./target/release/examples/detect_pipeline test_data/yolo_airport.jpg
```

**Expected Output:**
```
[INFO] Attempting to use CUDA backend (NVIDIA GPU)...
[INFO] ✓ Model loaded successfully with CUDA (NVIDIA GPU)
```

### Option 2: TensorRT (Maximum Performance)
TensorRT provides additional optimizations including:
- Layer fusion
- Kernel auto-tuning
- Dynamic tensor memory

```bash
# Install ONNX Runtime with TensorRT support (on Orin NX)
cargo build --release --features tensorrt --example detect_pipeline

# Run
./target/release/examples/detect_pipeline test_data/yolo_airport.jpg
```

**Expected Output:**
```
[INFO] Attempting to use TensorRT backend (NVIDIA GPU - optimized)...
[INFO] ✓ Model loaded successfully with TensorRT (NVIDIA GPU)
```

### Option 3: Both (Fallback Chain)
```bash
cargo build --release --features cuda,tensorrt --example detect_pipeline
```

**Fallback Order:**
1. Try CUDA first
2. If CUDA fails, try TensorRT
3. If both fail, fallback to CPU with FP32

## Prerequisites on Orin NX

### 1. Install JetPack SDK
```bash
# Check CUDA is installed
nvcc --version

# Should show CUDA 11.x or 12.x
```

### 2. Install ONNX Runtime for NVIDIA
```bash
# Download ONNX Runtime GPU build for Jetson
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-aarch64-gpu-1.17.0.tgz

# Extract and set library path
tar -xzf onnxruntime-linux-aarch64-gpu-1.17.0.tgz
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

### 3. Build with CUDA Support
```bash
# On Orin NX
cd /path/to/inference
cargo build --release --features cuda
```

## Performance Expectations

### FP32 (CPU Baseline)
- **Inference Time**: ~900ms per image (2 tiles, 640x640)
- **Memory**: 99MB model
- **Power**: ~5W

### FP16 (CUDA on Orin NX)
- **Inference Time**: ~350-450ms per image (2-2.5x faster) ⚡
- **Memory**: 50MB model (50% less)
- **Power**: ~8-10W
- **Batch Processing**: Excellent scaling with multiple tiles

### FP16 (TensorRT on Orin NX)
- **Inference Time**: ~250-350ms per image (2.5-3.5x faster) 🚀
- **Memory**: 50MB model + TensorRT engine cache
- **Power**: ~8-12W
- **First Run**: Slower due to engine building (~30s), then cached

## Verifying GPU Usage

```bash
# Monitor GPU utilization on Orin NX
sudo tegrastats

# You should see:
# - GPU utilization > 80%
# - EMC (memory controller) active
# - Power draw 8-12W during inference
```

## Troubleshooting

### CUDA Not Available
```
[WARN] CUDA initialization failed: ...
[INFO] Using CPU fallback model: ../models/yolov8m_batch_fp32.onnx
```

**Solutions:**
1. Check CUDA installation: `nvcc --version`
2. Verify ONNX Runtime GPU build is installed
3. Check library path: `echo $LD_LIBRARY_PATH`

### TensorRT Engine Building Takes Long
First run with TensorRT will take 20-60 seconds to build optimized engine. This is normal and only happens once. The engine is cached in:
```
~/.cache/onnxruntime/tensorrt_engines/
```

### Out of Memory
Orin NX has 8GB shared memory. If you get OOM errors:

```rust
let batch_config = BatchConfig {
    batch_size: 2,  // Reduce from 4
    timeout_ms: 50,
};
```

## Comparison: Apple M1 vs NVIDIA Orin NX

| Platform | Backend | Model | Inference Time | Speedup |
|----------|---------|-------|----------------|---------|
| **Apple M1 Max** | CoreML/Metal | FP32 | ~960ms | 1.0x |
| **Apple M1 Max** | CoreML/Metal | FP16* | Falls back to FP32 | 1.0x |
| **NVIDIA Orin NX** | CPU | FP32 | ~900ms | 1.0x |
| **NVIDIA Orin NX** | CUDA | FP16 | ~400ms | **2.2x** ⚡ |
| **NVIDIA Orin NX** | TensorRT | FP16 | ~300ms | **3.0x** 🚀 |

\* CoreML has type checking issues with FP16 ONNX models, but optimizes FP32 internally

## Recommended Configuration for Orin NX

```rust
// Orin NX optimized config
let detector_config = DetectorConfig {
    // Use FP16 model for NVIDIA GPU
    fp16_model_path: Some("models/yolov8m_batch_fp16.onnx".to_string()),
    fp32_model_path: Some("models/yolov8m_batch_fp32.onnx".to_string()),
    
    confidence_threshold: 0.22,
    nms_threshold: 0.45,
    input_size: (640, 640),
    
    // GPU settings
    use_gpu: true,
    gpu_device_id: 0,
    
    // CPU settings (for fallback)
    num_threads: Some(6),  // Orin NX has 8 cores, leave 2 for system
    
    ..Default::default()
};

let batch_config = BatchConfig {
    batch_size: 4,      // Orin NX can handle 4 tiles in parallel
    timeout_ms: 50,
};
```

## Summary

**For NVIDIA Orin NX, use FP16 model with CUDA or TensorRT for best performance:**

```bash
# Build command
cargo build --release --features cuda --example detect_pipeline

# Model config
fp16_model_path: "yolov8m_batch_fp16.onnx"  # 50MB, 2-3x faster! ⚡
```

**Expected Results:**
- ✅ **2-3x faster inference** compared to CPU
- ✅ **50% less memory bandwidth** (50MB vs 99MB)
- ✅ **Excellent batch processing** (multiple tiles in parallel)
- ✅ **Automatic fallback to CPU** if GPU unavailable
