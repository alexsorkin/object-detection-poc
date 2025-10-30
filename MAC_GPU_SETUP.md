# Mac GPU Training Setup Guide

This guide explains how to set up and optimize military target detection training on Mac with Apple Silicon GPU acceleration.

## Mac GPU Support Overview

### Apple Silicon Macs (M1/M2/M3/M4)
✅ **Full GPU acceleration support** via Metal Performance Shaders (MPS)
- **Performance**: 2-5x faster than CPU training
- **Memory**: Unified memory architecture (shared CPU/GPU memory)
- **Power Efficiency**: Excellent performance per watt

### Intel Macs
⚠️ **Limited support** - CPU training recommended
- Some models have discrete AMD GPUs but limited ML acceleration
- Better to use CPU training or consider cloud training

## Setup Instructions

### 1. System Requirements

**Minimum:**
- Mac with Apple Silicon (M1, M2, M3, or M4)
- macOS 12.3+ (for MPS support)
- 16GB unified memory (8GB minimum)
- 50GB free storage

**Recommended:**
- M2 Pro/Max or M3 Pro/Max/Ultra
- macOS 14+ (latest)
- 32GB+ unified memory
- 100GB free storage (for datasets)

### 2. Installation

```bash
# Clone and setup the project
git clone <repository>
cd military-target-detection

# Run the automated setup (includes Mac optimizations)
./setup.sh

# Activate the Python environment
source venv/bin/activate

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 3. Mac-Specific Configuration

The training system automatically detects Mac GPUs. Configuration in `training/config.yaml`:

```yaml
# Device configuration (auto-detects MPS on Mac)
device: "auto"  # Will use MPS if available

# Mac-optimized batch sizes
batch_size: 32  # M3 Pro/Max/Ultra
# batch_size: 16  # M1/M2 or lower memory
# batch_size: 8   # M1 with 8GB memory

# Worker processes (optimize for Mac)
workers: 8  # Good for most Apple Silicon Macs
```

### 4. Memory Optimization for Mac

Apple Silicon uses unified memory, so optimize accordingly:

```yaml
# For 8GB Macs
batch_size: 8
workers: 4
img_size: 416  # Reduce from 640 if needed

# For 16GB Macs  
batch_size: 16
workers: 6
img_size: 640

# For 32GB+ Macs
batch_size: 32
workers: 8
img_size: 640
```

### 5. Training Performance Tips

#### Memory Management
- **Monitor memory usage**: Activity Monitor > Memory tab
- **Close other apps** during training to free memory
- **Use smaller batch sizes** if you get memory errors
- **Enable swap** if you have limited memory

#### Performance Optimization
```python
# Add these optimizations to your training script
import torch

# Enable MPS optimizations
if torch.backends.mps.is_available():
    # Use memory-efficient attention (if available)
    torch.backends.mps.empty_cache()  # Clear cache between epochs
    
    # Set memory fraction (optional)
    # torch.mps.set_memory_fraction(0.8)  # Use 80% of memory
```

#### Thermal Management
- **Use a laptop stand** for better cooling
- **Monitor temperatures** with iStat Menus or similar
- **Lower batch size** if thermal throttling occurs
- **Train during cooler parts of the day**

### 6. Expected Performance

#### Training Speed (compared to CPU)
- **M1**: ~2-3x faster than CPU
- **M1 Pro/Max**: ~3-4x faster than CPU  
- **M2**: ~3-4x faster than CPU
- **M2 Pro/Max**: ~4-5x faster than CPU
- **M3 Pro/Max/Ultra**: ~5-8x faster than CPU

#### Typical Training Times (YOLOv8n, 100 epochs)
- **M1 (8GB)**: ~2-4 hours (small dataset)
- **M2 Pro (16GB)**: ~1-2 hours (medium dataset)
- **M3 Max (32GB+)**: ~30-60 minutes (large dataset)

### 7. Troubleshooting

#### Common Issues

**"MPS backend out of memory"**
```bash
# Solutions:
# 1. Reduce batch size
batch_size: 8  # or lower

# 2. Reduce image size
img_size: 416

# 3. Clear cache in training script
torch.backends.mps.empty_cache()
```

**"MPS not available"**
```bash
# Check macOS version
sw_vers

# Update to macOS 12.3+
# Or use CPU training:
device: "cpu"
```

**Slow training despite MPS**
```bash
# 1. Check thermal throttling
# 2. Close other applications
# 3. Verify MPS is actually being used:
python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
x = torch.tensor([1.]).to('mps')
print('MPS working:', x.device)
"
```

#### Performance Monitoring

```python
# Add to training script for monitoring
import psutil
import time

def log_system_stats():
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}% used ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu_percent}%")
    
    # MPS memory (if available)
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1e9
        cached = torch.mps.driver_allocated_memory() / 1e9
        print(f"MPS Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
```

### 8. Comparison with Other Platforms

| Platform | Training Speed | Setup Complexity | Cost |
|----------|---------------|------------------|------|
| Mac M3 Max | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| RTX 4090 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Google Colab | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AWS/GCP | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

### 9. Mac-Specific Training Commands

```bash
# Standard training with auto Mac GPU detection
python training/train.py --config config.yaml

# Force MPS device
python training/train.py --config config.yaml --device mps

# CPU fallback (if MPS issues)
python training/train.py --config config.yaml --device cpu

# Memory-optimized training for 8GB Macs
python training/train.py --config config_mac_8gb.yaml

# Performance monitoring
python training/train.py --config config.yaml --verbose
```

### 10. Mac-Optimized Model Export

```python
# Export with Mac optimizations
python training/export.py models/military_targets_best.pt \
    --formats onnx \
    --dynamic \
    --simplify \
    --opset 11 \
    --optimize-for-mobile  # Optional: for iOS deployment
```

## Conclusion

Mac Apple Silicon provides excellent performance for training military target detection models with proper configuration. The unified memory architecture and efficient MPS backend make it a competitive option for ML training, especially for development and medium-scale datasets.

For production training with very large datasets, consider cloud GPU instances, but Mac is perfect for:
- Development and prototyping
- Small to medium datasets
- Model fine-tuning
- Testing and validation
- Local development without cloud costs