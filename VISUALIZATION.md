# Detection Visualization Guide

## Overview

The `detect_yoloair` example now includes automatic visualization of detected objects with bounding boxes and labels.

## Usage

### Basic Usage (Default Output)

```bash
cd inference
cargo run --release --features metal --example detect_yoloair path/to/image.jpg
```

This will:
1. Load and process the image
2. Run detection with GPU acceleration
3. Draw bounding boxes with colored borders
4. Save the annotated image as `output_annotated.jpg`

### Custom Output Path

```bash
cargo run --release --features metal --example detect_yoloair input.jpg output.jpg
```

### Example

```bash
# Detect objects in an aerial photo
cargo run --release --features metal --example detect_yoloair ../test_images/aerial_view.jpg results/detected.jpg

# Output:
# 🎯 YOLO Aerial Detector - GPU Accelerated
# 
# ⚙️  Backend: ONNX Runtime (CoreML/Metal on macOS)
# 
# 📷 Loading image... ✓ (1920x1080)
# 📦 Loading model on GPU... ✓ (1.23s)
# 🚀 Running GPU inference... ✓ (180ms)
# 
# 📊 Results:
#   Detections: 3
#   Inference: 180ms (5.6 FPS)
# 
# 🎯 Detected Targets:
# 
#   Detection #1:
#     Class: person (ID: 0)
#     Confidence: 87.3%
#     Box: x=0.245, y=0.512, w=0.089, h=0.156
# 
#   Detection #2:
#     Class: car (ID: 2)
#     Confidence: 92.1%
#     Box: x=0.634, y=0.423, w=0.112, h=0.098
# 
#   Detection #3:
#     Class: truck (ID: 7)
#     Confidence: 78.5%
#     Box: x=0.123, y=0.678, w=0.145, h=0.134
# 
# 🎨 Drawing bounding boxes...
#   ✓ Drew box for: person at (470, 553) size 171x169
#   ✓ Drew box for: car at (1217, 457) size 215x106
#   ✓ Drew box for: truck at (236, 732) size 278x145
# 
# ✓ Drawing complete
# 💾 Saved annotated image to: results/detected.jpg
```

## Visualization Features

### Bounding Boxes
- **Color**: Each class gets a unique, deterministic color
- **Thickness**: 3-pixel thick borders for high visibility
- **Coordinates**: Automatically converted from normalized (0-1) to pixel coordinates

### Class Labels
- **Label Bar**: Colored rectangle above each detection
- **Content**: Shows class name and confidence percentage
- **Colors**: Match the bounding box color for easy association

### Color Scheme

Colors are automatically generated using the golden angle algorithm for optimal distribution:

- **Person** (ID 0): Red tones
- **Bicycle** (ID 1): Orange tones  
- **Car** (ID 2): Yellow-green tones
- **Motorcycle** (ID 3): Green tones
- **Airplane** (ID 4): Cyan tones
- **Clock** (ID 74): Purple tones
- And so on...

Each class ID gets a consistent, visually distinct color.

## Detection Information

The example outputs:

1. **Image Info**: Resolution and loading time
2. **Model Info**: GPU loading time
3. **Inference Stats**: Processing time and FPS
4. **Detection List**: All detected objects with:
   - Class name and ID
   - Confidence percentage
   - Normalized bounding box coordinates
5. **Visualization**: Pixel coordinates of drawn boxes
6. **Output**: Path to saved annotated image

## Tips

### For Aerial Images

```bash
# Use lower confidence threshold for distant objects
# Edit the example to set: confidence_threshold: 0.10
cargo run --release --features metal --example detect_yoloair aerial.jpg output.jpg
```

### For High-Resolution Images

The visualization automatically scales to any image size. Bounding boxes are drawn at full resolution.

### Batch Processing

Create a simple script to process multiple images:

```bash
#!/bin/bash
for img in images/*.jpg; do
    output="results/$(basename $img)"
    cargo run --release --features metal --example detect_yoloair "$img" "$output"
done
```

## Supported Models

The visualization works with any YOLO model format:

- **COCO models** (80 classes): person, car, dog, clock, etc.
- **Objects365 models** (365 classes): Extended object set
- **Custom models**: Any class IDs, labeled as `class_N`
- **Single-class models**: Aerial detectors, specialized models

## Performance

- **Inference**: 180ms (mini) / 496ms (full) on AMD Radeon Pro 5500M
- **Drawing**: ~50-100ms for typical image with 10-20 detections
- **Saving**: ~100-200ms depending on image size

Total time: ~330-800ms per image (including I/O)

## Advanced: Adding Text Labels

For production use, you can add proper text rendering:

1. Add `rusttype` to `Cargo.toml`:
```toml
rusttype = "0.9"
```

2. Include a font file in `assets/DejaVuSans.ttf`

3. Use `draw_text_mut()` from `imageproc` to render text labels

See the commented code in the example for reference.

## Troubleshooting

### Issue: Image not found
```
Error: No such file or directory (os error 2)
```
**Solution**: Check the image path is correct. Use absolute paths if needed.

### Issue: Model not found
```
Error: Failed to load model
```
**Solution**: Ensure the model path in the example points to your ONNX model file.

### Issue: No detections
```
ℹ️  No targets detected
```
**Solution**: Try lowering the `confidence_threshold` in the example code (default: 0.10).

### Issue: Out of memory
```
Error: Failed to allocate
```
**Solution**: Use the mini model or resize large images before processing.

## Examples

### Military Target Detection
```bash
cargo run --release --features metal --example detect_yoloair battlefield.jpg result.jpg
```

### Aerial Vehicle Detection
```bash
cargo run --release --features metal --example detect_yoloair drone_footage.jpg detected_vehicles.jpg
```

### General Object Detection (COCO)
```bash
cargo run --release --features metal --example detect_yoloair street_scene.jpg annotated.jpg
```

## Output Format

The annotated image is saved as a standard JPEG file with:
- Original image as background
- Colored bounding boxes overlaid
- Label bars with class names
- High-resolution, ready for analysis or presentation

You can open the output with any image viewer or use it in reports, documentation, or further processing pipelines.
