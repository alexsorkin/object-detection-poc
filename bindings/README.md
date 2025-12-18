# Unity C# Bindings for Military Target Detection

## Overview

This directory contains Unity C# bindings for the military target detection library built with RT-DETR and ONNX Runtime.

## Files

- `MilitaryTargetDetector.cs` - Main Unity component wrapper
- `RealtimeDetectionExample.cs` - Example implementation for real-time detection
- `README.md` - This documentation file

## Key Features

- **RT-DETR Detection**: State-of-the-art real-time detection transformer
- **COCO Classes**: 80 standard object classes supported
- **GPU Acceleration**: Compatible with CUDA, TensorRT, CoreML, OpenVINO, ROCm
- **Unity Integration**: Direct integration as Unity MonoBehaviour component

## API Changes (Updated December 2025)

### Breaking Changes from Previous Version:
1. **Constructor Parameters**: Added `detector_type` parameter (use `0` for RT-DETR)
2. **Removed NMS Threshold**: NMS is now handled internally by the model
3. **Class Definitions**: Updated to use standard COCO 80-class definitions
4. **Configuration Updates**: Runtime configuration changes require detector recreation

### Supported COCO Classes (80 total):
- Person, Vehicle, Animals, Household objects, Sports equipment, etc.
- See `TargetClass` enum in `MilitaryTargetDetector.cs` for complete list

## Usage Example

```csharp
// Basic setup
var detector = GetComponent<MilitaryTargetDetector>();
detector.modelPath = "rtdetr_v2_r50vd_fp32.onnx";
detector.confidenceThreshold = 0.5f;
detector.maxDetections = 100;
detector.useGPU = true;

// Initialize
if (detector.InitializeDetector())
{
    // Detect from texture
    var result = detector.DetectObjects(inputTexture);
    
    // Process results
    foreach (var detection in result.detections)
    {
        Debug.Log($"Detected {detection.targetClass} with confidence {detection.confidence}");
    }
}
```

## GPU Backend Support

The bindings support multiple GPU acceleration backends:

- **CUDA**: NVIDIA GPUs (RTX series, etc.)
- **TensorRT**: Optimized NVIDIA inference
- **CoreML**: Apple Silicon (M1/M2/M3 Macs)
- **OpenVINO**: Intel GPUs and VPUs
- **ROCm**: AMD GPUs

## Performance Notes

- **GPU Required**: For real-time performance (>20 FPS), GPU acceleration is essential
- **Model Size**: FP16 models provide better performance with minimal accuracy loss
- **Input Resolution**: 640x640 is recommended for best speed/accuracy balance
- **Warmup**: First inference may be slower due to GPU initialization

## Integration Steps

1. **Copy Files**: Place both `.cs` files in your Unity project's `Scripts` folder
2. **Build Native Library**: Compile the Rust library as a shared library (.dll/.so/.dylib)
3. **Unity Setup**: 
   - Place native library in `Assets/Plugins/`
   - Place ONNX model files in `Assets/StreamingAssets/`
4. **Component Setup**: Add `MilitaryTargetDetector` component to a GameObject
5. **Configuration**: Set model path, confidence threshold, and GPU settings in inspector

## Troubleshooting

### Common Issues:
- **"Library not found"**: Ensure native library is in correct Plugins folder for target platform
- **"Model not found"**: Verify ONNX model is in StreamingAssets folder
- **Poor Performance**: Enable GPU acceleration and use FP16 models
- **No Detections**: Check confidence threshold and input image format

### Debug Information:
```csharp
// Enable logging
detector.enableLogging = true;

// Check initialization
if (!detector.InitializeDetector()) 
{
    Debug.LogError("Failed to initialize detector");
}

// Verify GPU usage
Debug.Log($"Using GPU: {detector.useGPU}");
```

## Platform Support

- **Windows**: CUDA, TensorRT, OpenVINO
- **macOS**: CoreML (Apple Silicon)  
- **Linux**: CUDA, TensorRT, OpenVINO, ROCm

## Requirements

- Unity 2022.3 LTS or newer
- Native library compiled for target platform
- GPU drivers for chosen acceleration backend
- RT-DETR ONNX model files

## Performance Benchmarks

Typical performance on RTX 5080:
- **CUDA**: ~14ms per frame (70+ FPS)
- **TensorRT**: ~10ms per frame (100+ FPS) 
- **CPU Fallback**: ~500ms per frame (2 FPS)

## License

Part of the Military Target Detection project. See main project README for license details.