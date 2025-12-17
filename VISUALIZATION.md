# Detection Visualization Guide

## Overview

The RT-DETR detection pipeline includes automatic visualization of detected objects with bounding boxes and labels.

## Usage

### Video Detection with Real-time Visualization

```bash
cd inference
cargo run --release --features metal --example detect_video path/to/video.mp4
```

This will:
1. Load and process the video
2. Run detection with GPU acceleration
3. Draw bounding boxes with colored borders
4. Display results in real-time window
5. Save annotated video as `output_video.mp4`

### Custom Configuration

```bash
cargo run --release --features metal --example detect_video -- \
  --confidence 60 \
  --classes 0,2,3,4,7 \
  --model r50_fp32 \
  test_data/airport.mp4
```

### Example Output

```bash
# Detect objects in video
cargo run --release --features metal --example detect_video test_data/airport.mp4

# Output:
# 📝 Parsing arguments...
# 🤖 Using detector: RTDETR
# 📹 Opening video source: test_data/airport.mp4... ✓
# 
# 💡 Configuration:
#   • Detector: rtdetr_v2_r50vd_fp32 (frame executor)
#   • Confidence: 50%
#   • Classes: [0, 2, 3, 4, 7]
#   • Tracker: ByteTrack
# 
# 🎬 Starting video processing...
#   Press 'q' to quit, 'p' to pause
# 
# 📊 Detection Statistics:
#   Detections: 156
#   Inference: 45ms (22.2 FPS)
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

### Confidence Thresholds

```bash
# Use lower confidence threshold for distant objects
cargo run --release --features metal --example detect_video -- --confidence 30 test_data/video.mp4
```

### High-Resolution Videos

The visualization automatically scales to any video size. Bounding boxes are drawn at full resolution.

### Headless Mode

For processing without display:

```bash
cargo run --release --features metal --example detect_video -- --headless test_data/video.mp4
```

## Supported Models

The visualization works with RT-DETR models:

- **RT-DETR r18**: Fast, suitable for real-time applications
- **RT-DETR r34/r50**: More accurate, higher latency
- **RT-DETR r101**: Maximum accuracy
- **FP16 variants**: GPU-accelerated inference
- **Custom trained models**: Any class IDs supported
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

### Issue: Video not found
```
Error: No such file or directory (os error 2)
```
**Solution**: Check the video path is correct. Use absolute paths if needed.

### Issue: Model not found
```
Error: Failed to load model
```
**Solution**: Ensure the RT-DETR model files are in the `models/` directory.

### Issue: No detections
```
No detections found
```
**Solution**: Try lowering the confidence threshold: `--confidence 30`

### Issue: Low FPS
```
Slow detection performance
```
**Solution**: Use FP16 models for GPU acceleration or lower resolution videos.

## Examples

### Airport Security
```bash
cargo run --release --features metal --example detect_video test_data/airport.mp4
```

### Traffic Monitoring
```bash
cargo run --release --features metal --example detect_video -- --classes 2,3,4,7 traffic.mp4
```

### Real-time Webcam
```bash
cargo run --release --features metal --example detect_video 0
```

## Output Format

The annotated video is saved as `output_video.mp4` with:
- Original video frames as background
- Colored bounding boxes overlaid
- Real-time tracking IDs
- Green boxes for detections, yellow for tracker predictions
- 30 FPS output with smooth tracking

You can open the output with any video player or use it for analysis, documentation, or further processing.
