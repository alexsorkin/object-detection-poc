# CUDA Setup for ONNX Runtime

## Current Status

✅ **Working:**
- NVIDIA GPU detected: RTX 5080 (16GB)
- CUDA 12.0/13.0 installed
- CUDA toolkit installed
- ONNX Runtime CUDA provider downloaded
- Cargo feature `cuda` compiles successfully

❌ **Missing:**
- **cuDNN 9** (CUDA Deep Neural Network library)

## Error Diagnosis

When checking the CUDA provider library:
```bash
$ ldd target/debug/libonnxruntime_providers_cuda.so
libcudnn.so.9 => not found  # <-- THIS IS THE PROBLEM
```

The ONNX Runtime CUDA provider requires cuDNN to be installed on the system.

## Installation Steps

### Option 1: Install cuDNN from NVIDIA (Recommended)

1. **Register for NVIDIA Developer account**:
   - Go to https://developer.nvidia.com/cudnn
   - Create a free account

2. **Download cuDNN**:
   - Download cuDNN 9.x for CUDA 12.x
   - Choose: "cuDNN Library for Linux (x86_64)"

3. **Install cuDNN**:
   ```bash
   # Extract the downloaded archive
   tar -xvf cudnn-linux-x86_64-9.*.tar.xz
   
   # Copy files to CUDA installation
   sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
   sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   
   # Update library cache
   sudo ldconfig
   ```

### Option 2: Try Ubuntu Package (May not have cuDNN 9)

```bash
# Search for available cuDNN packages
apt-cache search libcudnn

# Install if available (adjust version as needed)
sudo apt-get update
sudo apt-get install libcudnn9 libcudnn9-dev

# For CUDA 12 specifically:
sudo apt-get install libcudnn9-cuda-12
```

### Option 3: Use Docker with CUDA/cuDNN Pre-installed

```bash
docker run --gpus all -it nvidia/cuda:12.0.0-cudnn9-devel-ubuntu24.04 bash
```

## Verification

After installing cuDNN, verify the installation:

```bash
# Check if cuDNN is found
ldconfig -p | grep cudnn

# Should show something like:
# libcudnn.so.9 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudnn.so.9
```

Then rebuild and test:

```bash
cd inference
cargo build --features cuda --example test_cuda
cargo run --features cuda --example test_cuda
```

## Check CUDA Usage at Runtime

To verify CUDA is actually being used when running detection:

```bash
# Terminal 1: Monitor GPU usage
watch -n 0.5 nvidia-smi

# Terminal 2: Run detection with CUDA
RUST_LOG=info cargo run --features cuda --example detect_video -- input.mp4
```

Look for log messages like:
```
INFO  RT-DETR loaded successfully with CUDA (NVIDIA GPU)
```

You should see GPU utilization increase in `nvidia-smi`.

## Alternative: CPU-only Mode

If you can't install cuDNN immediately, the code will automatically fall back to CPU:

```bash
# Build without CUDA feature (uses CPU)
cargo build --release
cargo run --release --example detect_video -- input.mp4
```

The detector will log:
```
WARN  CUDA initialization failed: ...
INFO  Using CPU fallback model
INFO  RT-DETR loaded successfully with CPU
```

## ONNX Runtime Details

The `ort` crate automatically downloads ONNX Runtime with CUDA support when you enable the `cuda` feature:

- **Cache location**: `~/.cache/ort.pyke.io/`
- **CUDA provider**: `libonnxruntime_providers_cuda.so`
- **TensorRT provider**: `libonnxruntime_providers_tensorrt.so` (also available)

These are symlinked to your `target/debug/` directory during build.

## System Requirements

For CUDA acceleration you need:
1. ✅ NVIDIA GPU (compute capability 6.0+)
2. ✅ CUDA Toolkit 11.x or 12.x
3. ❌ **cuDNN 8.x or 9.x** (MISSING - install this!)
4. ✅ NVIDIA driver 450.80.02 or higher

## Troubleshooting

### CUDA feature compiles but doesn't use GPU

**Symptom**: Build succeeds but runs on CPU
**Cause**: cuDNN library not found at runtime
**Solution**: Install cuDNN (see above)

### "CUDA initialization failed" at runtime

Check the full error message:
```bash
RUST_LOG=debug cargo run --features cuda --example detect_video
```

Common issues:
- `libcudnn.so.9 not found` → Install cuDNN
- `CUDA out of memory` → Reduce batch size or use smaller model
- `Unsupported GPU` → Check compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`

### Verify CUDA is working

```bash
# Check CUDA samples
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# Should show your RTX 5080 with CUDA capability 9.0+
```
