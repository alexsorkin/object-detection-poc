# Military Target Detection Examples

This directory contains example applications and scripts demonstrating how to use the military target detection system across different platforms.

## Examples Overview

### Python Training Examples
- `train_example.py` - Complete training workflow
- `export_example.py` - Model export to different formats
- `validation_example.py` - Model validation and testing

### Rust Library Examples
- `basic_detection.rs` - Simple image detection
- `video_processing.rs` - Real-time video processing
- `batch_processing.rs` - Batch image processing
- `performance_benchmark.rs` - Performance testing

### Unity Integration Examples
- `UnityRealtimeDetection/` - Complete Unity project
- `VRDetectionExample/` - VR/AR specific implementation
- `MobileOptimized/` - Mobile-optimized version

### Command-Line Tools
- `detect_cli.rs` - Command-line detection tool
- `video_cli.rs` - Video processing tool
- `benchmark_cli.rs` - Performance benchmarking

## Usage Instructions

### Training a Model

1. **Prepare your dataset** in the required YOLO format:
   ```
   data/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

2. **Run training**:
   ```bash
   cd training
   python train.py --config config.yaml
   ```

3. **Export model**:
   ```bash
   python export.py models/military_targets_best.pt --formats onnx
   ```

### Using Rust Library

1. **Basic detection**:
   ```bash
   cd inference
   cargo run --example basic_detection -- --model ../models/military_targets.onnx --image test.jpg
   ```

2. **Video processing**:
   ```bash
   cargo run --features opencv --example video_processing -- --model ../models/military_targets.onnx --camera 0
   ```

### Unity Integration

1. **Copy files** to your Unity project:
   - Copy `bindings/MilitaryTargetDetector.cs` to `Assets/Scripts/`
   - Copy the compiled Rust library to `Assets/Plugins/`
   - Copy the ONNX model to `Assets/StreamingAssets/Models/`

2. **Add component** to a GameObject in your scene
3. **Configure** model path and detection parameters
4. **Start detection** via the component or script

### VR/AR Applications

The system is optimized for VR/AR with:
- Low-latency inference (<33ms per frame)
- Mobile GPU acceleration
- Memory-efficient processing
- Unity XR toolkit integration

## Performance Optimization Tips

### Model Optimization
- Use YOLOv8n (nano) for mobile/VR devices
- Use YOLOv8s or YOLOv8m for desktop applications
- Enable GPU acceleration when available
- Use TensorRT for NVIDIA GPUs

### Memory Management
- Process frames at 640x640 resolution for best speed/accuracy trade-off
- Use batch processing for multiple streams
- Enable model quantization for mobile devices

### Real-time Processing
- Target 30 FPS for smooth real-time operation
- Use separate threads for capture and processing
- Implement frame dropping for consistent frame rate
- Monitor CPU/GPU usage and adjust accordingly

## Deployment Considerations

### Security
- This system is designed for defensive applications only
- Ensure proper access controls and data handling
- Follow local regulations regarding surveillance systems
- Implement proper logging and audit trails

### Privacy
- Respect privacy rights and obtain necessary permissions
- Use only legally obtained training data
- Implement data anonymization when required
- Follow GDPR and other privacy regulations

### Hardware Requirements

#### Minimum (CPU-only):
- Intel i5 or AMD Ryzen 5 processor
- 8GB RAM
- 100MB disk space for model

#### Recommended (GPU-accelerated):
- Intel i7 or AMD Ryzen 7 processor
- 16GB RAM
- NVIDIA GTX 1060 / RTX 2060 or better
- 500MB disk space

#### Mobile/VR:
- ARM Cortex-A78 or better
- 6GB RAM
- Adreno 660 / Mali-G78 or better
- 200MB storage space

## Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Check file path is correct
   - Ensure ONNX Runtime is properly installed
   - Verify model file isn't corrupted

2. **Poor detection accuracy**:
   - Check input image quality and lighting
   - Adjust confidence threshold
   - Ensure model was trained on similar data

3. **Slow inference**:
   - Enable GPU acceleration
   - Reduce input image size
   - Use optimized model (TensorRT, quantized)

4. **Unity integration issues**:
   - Ensure library is in correct platform folder
   - Check Unity's native plugin settings
   - Verify model path in StreamingAssets

### Performance Issues

1. **Low FPS**:
   - Reduce detection frequency
   - Use smaller input size
   - Enable GPU processing
   - Implement frame skipping

2. **High memory usage**:
   - Reduce batch size
   - Clear detection results regularly
   - Use streaming for large videos

3. **GPU out of memory**:
   - Reduce input size
   - Decrease batch size
   - Use model quantization

## Support and Contributing

For issues, feature requests, or contributions:
1. Check existing documentation and examples
2. Review troubleshooting section
3. Create detailed issue reports with:
   - System specifications
   - Error messages
   - Steps to reproduce
   - Expected vs actual behavior

## License and Legal

This software is provided for research and defensive applications only. Users are responsible for:
- Complying with local laws and regulations
- Obtaining proper permissions for data collection
- Ensuring ethical use of detection capabilities
- Respecting privacy rights and civil liberties

The detection system should not be used for:
- Offensive military operations
- Unauthorized surveillance
- Privacy violations
- Discrimination or profiling