# Military Target Detector - Benchmarks

Comprehensive benchmarking suite for measuring and optimizing performance of the military target detection library.

## 🎯 Overview

This benchmarking suite provides detailed performance analysis across all core components:

- **🖼️ Image Processing**: Drawing, annotation, and image manipulation operations
- **🎯 Tracking**: Multi-object tracking algorithms (Kalman, ByteTrack)  
- **🤖 Detection**: Neural network inference performance (RT-DETR models)
- **🔄 Pipelines**: Complete system integration and throughput

## 🚀 Quick Start

### Run All Benchmarks
```bash
# Navigate to the inference directory
cd /path/to/model/inference

# Run comprehensive benchmark suite
./benches/run_benchmarks.sh
```

### Run Individual Benchmarks
```bash
# Image processing (fastest, no dependencies)
cargo bench --bench image_processing

# Tracking algorithms (fast, no model dependencies)  
cargo bench --bench object_tracking

# Detection models (requires ONNX models)
cargo bench --bench rtdetr_detection

# Complete pipelines (slowest, requires models)
cargo bench --bench frame_pipeline
```

## 📊 Benchmark Categories

### 1. Image Processing Benchmarks (`image_processing.rs`)

Tests core image manipulation performance:

- **Scale Factor Calculation**: Coordinate transformation math
- **Annotation Preparation**: Parallel processing of detection data
- **Batch Drawing**: Rectangle and text rendering optimizations
- **Complete Pipeline**: End-to-end annotation workflow
- **Resolution Impact**: Performance scaling with image size

**Key Metrics**: Operations/second, memory usage, scaling efficiency

### 2. Tracking Benchmarks (`object_tracking.rs`)

Evaluates multi-object tracking algorithms:

- **Kalman Filter**: Classic tracking with Hungarian data association
- **ByteTrack**: Modern algorithm with occlusion handling
- **Algorithm Comparison**: Head-to-head performance analysis
- **Sequence Tracking**: Multi-frame temporal performance
- **Parameter Sensitivity**: IoU threshold and configuration impact

**Key Metrics**: Frames/second, tracking accuracy, memory efficiency

### 3. Detection Benchmarks (`rtdetr_detection.rs`)

Measures neural network inference performance:

- **Resolution Scaling**: Performance across VGA to 4K resolutions
- **Batch Processing**: Throughput optimization for multiple frames
- **Model Variants**: R18/R34/R50 performance comparison
- **Configuration Impact**: Confidence threshold and parameter effects

**Key Metrics**: Inference time, throughput, accuracy trade-offs

**Requirements**: RT-DETR ONNX models in `../models/` directory

### 4. Pipeline Benchmarks (`frame_pipeline.rs`)

Tests complete system integration:

- **Frame Executor**: Async detection pipeline performance
- **Detection Pipeline**: End-to-end processing workflow
- **Video Pipeline**: Real-time video processing with tracking
- **Submission Rates**: Frame ingestion and backpressure handling
- **Queue Management**: Buffer sizing and flow control

**Key Metrics**: End-to-end latency, throughput, system resource usage

**Requirements**: RT-DETR ONNX models and full system setup

## 📁 File Structure

```
benches/
├── rtdetr_detection.rs             # Neural network inference tests
├── object_tracking.rs              # Multi-object tracking tests  
├── image_processing.rs             # Image manipulation tests
├── frame_pipeline.rs               # System integration tests
├── run_benchmarks.sh              # Comprehensive runner script
└── README.md                      # This documentation
```

## 🔧 Configuration

### Environment Variables

- `DEFENITY_MODEL_DIR`: Path to ONNX model directory (default: `../models`)
- `RUST_LOG`: Log level for benchmarking (recommend: `warn`)

### Model Requirements

For full benchmark coverage, ensure these models are available:

- `rtdetr_v2_r18vd_fp32.onnx` - Baseline performance model
- `rtdetr_v2_r34vd_fp32.onnx` - Balanced model (optional)
- `rtdetr_v2_r50vd_fp32.onnx` - High accuracy model (optional)

### System Recommendations

- **CPU**: Multi-core processor for parallel operations
- **Memory**: 8GB+ RAM for large image processing
- **Storage**: SSD recommended for model loading
- **Network**: Isolated system for consistent measurements

## 📈 Interpreting Results

### Performance Metrics

- **Throughput**: Elements/operations per second
- **Latency**: Time per individual operation
- **Memory**: Peak memory usage during operations
- **Scaling**: Performance change with input size

### Result Analysis

1. **Baseline Establishment**: Run benchmarks on clean system
2. **Optimization Testing**: Compare before/after changes
3. **Configuration Tuning**: Find optimal parameters
4. **Regression Detection**: Monitor for performance degradation

### HTML Reports

Benchmarks generate detailed HTML reports with:

- Interactive charts and graphs
- Statistical analysis (mean, median, std dev)
- Historical comparison tracking  
- Performance regression detection

## 🎛️ Benchmark Parameters

### Detection Benchmarks

- **Image Sizes**: 640x480, 1280x720, 1920x1080, 2560x1440
- **Batch Sizes**: 1, 2, 4, 8 frames
- **Confidence Thresholds**: 0.1, 0.3, 0.5, 0.7, 0.9
- **Model Variants**: R18, R34, R50 (FP32)

### Tracking Benchmarks

- **Object Counts**: 5, 10, 25, 50 objects per frame
- **Sequence Lengths**: 10, 30, 60 frames
- **IoU Thresholds**: 0.1, 0.3, 0.5, 0.7
- **Tracking Methods**: Kalman, ByteTrack

### Image Processing Benchmarks

- **Resolutions**: VGA to 4K
- **Object Counts**: 1 to 100 annotations per frame
- **Batch Sizes**: Various annotation densities
- **Pipeline Loads**: Light, medium, heavy, extreme

### Pipeline Benchmarks

- **Queue Depths**: 1, 2, 5 frames
- **Frame Rates**: 30 FPS simulation
- **Tracking Configs**: Kalman vs ByteTrack
- **Backpressure**: Overload scenario testing

## 🔍 Troubleshooting

### Common Issues

**Models Not Found**
```bash
export DEFENITY_MODEL_DIR=/path/to/models
./benches/run_benchmarks.sh
```

**Benchmark Timeout**
```bash
# Increase timeout for slow systems
cargo bench --bench rtdetr_detection -- --measurement-time 30
```

**Memory Issues**
- Reduce batch sizes in benchmarks
- Monitor system memory usage
- Close other applications

**Inconsistent Results**
- Run on isolated system
- Disable CPU frequency scaling
- Use consistent power settings

### Debug Mode

```bash
# Enable detailed logging
RUST_LOG=debug cargo bench --bench object_tracking

# Single benchmark iteration
cargo bench --bench image_processing -- --sample-size 10
```

## 🤝 Contributing

### Adding New Benchmarks

1. Create benchmark functions following existing patterns
2. Use realistic test data representative of production usage
3. Include multiple parameter variations (sizes, counts, etc.)
4. Add appropriate throughput measurements
5. Document expected performance characteristics

### Best Practices

- **Realistic Data**: Use representative input sizes and patterns
- **Multiple Scenarios**: Test various configurations and edge cases  
- **Statistical Validity**: Ensure sufficient sample sizes
- **Resource Monitoring**: Track memory and CPU usage
- **Documentation**: Explain benchmark purpose and interpretation

### Benchmark Guidelines

- Focus on critical path performance
- Test both average and worst-case scenarios
- Include parameter sensitivity analysis
- Validate against production workloads
- Maintain benchmark stability across code changes

## 📚 Further Reading

- [Criterion.rs Documentation](https://docs.rs/criterion/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Military Target Detector Documentation](../README.md)
- [RT-DETR Model Performance Guide](../docs/RTDETR_IMPLEMENTATION.md)