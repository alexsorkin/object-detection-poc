# Test Image Normalization

## Image Sizes

### Original Image
- **File**: `test_data/yolo_air_test.jpg`
- **Size**: 1120x787 pixels (726 KB)
- **Aspect ratio**: ~1.42:1

### Normalized Image
- **File**: `test_data/yolo_air_test_640.jpg`  
- **Size**: 640x640 pixels (157 KB)
- **Aspect ratio**: 1:1 (square, YOLO standard)
- **Scaling**: 640x450 content + 95px gray padding top/bottom

## Why Normalize?

YOLO models expect **640x640** square inputs. The preprocessing pipeline:

1. **Without normalization** (using original 1120x787):
   - Detector loads 1120x787 image
   - Resizes to 640x450 (maintaining aspect)
   - Adds gray padding to make 640x640
   - **Memory**: ~2.6 MB in memory
   - **Time**: ~5-10ms extra preprocessing

2. **With normalization** (using 640x640):
   - Detector loads 640x640 image (already optimal)
   - No resizing needed
   - **Memory**: ~1.2 MB in memory
   - **Time**: Minimal preprocessing

## Benefits of Normalized Images

✅ **Faster inference**: Skip resize operation  
✅ **Lower memory**: 50% less memory usage  
✅ **Consistent**: Same preprocessing as training  
✅ **Predictable**: No aspect ratio surprises

## How to Normalize

Run the normalization script:

```bash
cd /Users/alexandersorkin/defenity/model
python3 scripts/normalize_test_image.py
```

Output:
```
YOLO Image Normalization
==================================================
Input:  test_data/yolo_air_test.jpg
Output: test_data/yolo_air_test_640.jpg
Target: 640x640 (YOLOv8 standard input size)

Normalizing image: test_data/yolo_air_test.jpg
  Original size: 1120x787
  Scaled size: 640x450
  ✓ Saved to: test_data/yolo_air_test_640.jpg
  Final size: 640x640

✓ Normalization complete!
```

## Testing

### With Original Image
```bash
cd inference
cargo run --release --features metal --example detect_yoloair test_data/yolo_air_test.jpg output_original.jpg
```

### With Normalized Image (Recommended)
```bash
cd inference
cargo run --release --features metal --example detect_yoloair test_data/yolo_air_test_640.jpg output_normalized.jpg
```

Both will produce the same detection results, but the normalized version is slightly faster and uses less memory.

## Normalization Details

The script:
1. Loads the original image (1120x787)
2. Scales it down maintaining aspect ratio to fit in 640x640
3. Centers the scaled image (640x450) in a 640x640 canvas
4. Fills padding with gray color (114, 114, 114) - same as YOLO training
5. Saves as JPEG with 95% quality

Result:
```
┌─────────────────────────────────────┐
│  Gray padding (95px top)            │
├─────────────────────────────────────┤
│                                     │
│  Scaled image content (640x450)    │
│                                     │
├─────────────────────────────────────┤
│  Gray padding (95px bottom)         │
└─────────────────────────────────────┘
         640x640 total
```

## Available Test Images

All images in `inference/test_data/`:

1. **yolo_air_test.jpg** (726 KB, 1120x787)
   - Original high-resolution aerial image
   - Good for quality testing
   
2. **yolo_air_test_640.jpg** (157 KB, 640x640) ✨
   - Normalized for YOLO
   - **Recommended for testing**
   - Optimal performance

3. **test_tank_detection.jpg** (144 KB)
   - Military tank detection test
   
4. **test_tank.jpg** (139 KB)
   - Tank test image

## Recommendation

For production use and testing, always use **640x640 normalized images** when possible:

- Faster inference
- Lower memory usage  
- More predictable results
- Matches training conditions

The detector handles any size automatically, but normalized images are optimal! 🎯
