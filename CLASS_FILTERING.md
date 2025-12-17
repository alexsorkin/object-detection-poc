# Class Filtering Guide

## Overview

The detector can filter detections to only show specific object classes. This is useful for aerial vehicle detection, traffic monitoring, or any scenario where you only care about certain object types.

## Default Configuration

The example is configured to detect only:
- ✈️ **Airplane** (COCO class ID 4)
- 🚗 **Car** (COCO class ID 2)  
- 🚚 **Truck** (COCO class ID 7)

## How It Works

```rust
// Define allowed classes
let allowed_classes = vec![
    "airplane",
    "car", 
    "truck",
];

// Filter detections
let detections: Vec<_> = detections
    .into_iter()
    .filter(|det| {
        let class_name = det.class.name();
        allowed_classes.contains(&class_name.as_str())
    })
    .collect();
```

The filter:
1. Runs full detection on the image
2. Removes detections that don't match allowed classes
3. Only shows/draws the filtered results

## Example Output

```
🚀 Running GPU inference... ✓ (180ms)

📊 Results:
  Total detections: 15
  Filtered out: 12 (other classes)
  Matching classes: 3
  Allowed: airplane, car, truck
  Inference: 180ms (5.6 FPS)

🎯 Detected Targets:

  Detection #1:
    Class: airplane (ID: 4)
    Confidence: 87.3%
    Box: x=0.245, y=0.512, w=0.089, h=0.156

  Detection #2:
    Class: car (ID: 2)
    Confidence: 92.1%
    Box: x=0.634, y=0.423, w=0.112, h=0.098

  Detection #3:
    Class: truck (ID: 7)
    Confidence: 78.5%
    Box: x=0.123, y=0.678, w=0.145, h=0.134
```

## Common Filter Configurations

### Aerial Vehicles Only
```rust
let allowed_classes = vec![
    "airplane",
    "helicopter",  // If model supports it
];
```

### All Vehicles
```rust
let allowed_classes = vec![
    "airplane",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "boat",
];
```

### Traffic Monitoring
```rust
let allowed_classes = vec![
    "car",
    "truck",
    "bus",
    "motorcycle",
];
```

### Parking Lot Analysis
```rust
let allowed_classes = vec![
    "car",
    "truck",
    "motorcycle",
];
```

### People Detection
```rust
let allowed_classes = vec![
    "person",
];
```

### Military Targets (Custom Model)
```rust
let allowed_classes = vec![
    "class_0",  // Armed Personnel
    "class_1",  // Rocket Launcher
    "class_2",  // Military Vehicle
    "class_3",  // Heavy Weapon
];
```

## COCO Class Reference

Common classes for aerial/vehicle detection:

| Class Name | COCO ID | Use Case |
|------------|---------|----------|
| `person` | 0 | People detection |
| `bicycle` | 1 | Cyclists |
| `car` | 2 | Cars, sedans |
| `motorcycle` | 3 | Motorcycles |
| `airplane` | 4 | Aircraft |
| `bus` | 5 | Buses |
| `train` | 6 | Trains |
| `truck` | 7 | Trucks, pickups |
| `boat` | 8 | Boats, ships |
| `traffic light` | 9 | Traffic signals |
| `fire hydrant` | 10 | Fire hydrants |
| `stop sign` | 11 | Stop signs |

Full list: See [CLASS_SUPPORT.md](../CLASS_SUPPORT.md) for all 80 COCO classes.

## Objects365 Classes

If using Objects365 model, additional vehicle classes:

```rust
let allowed_classes = vec![
    "suv",
    "van", 
    "pickup truck",
    "sports car",
    "race car",
    "machinery vehicle",
    "fire truck",
    "ambulance",
    "sailboat",
    "helicopter",
];
```

## Performance Impact

Class filtering happens **after inference**, so it:

✅ **No impact on inference speed** - Filtering is instant  
✅ **Reduces output size** - Fewer detections to process  
✅ **Cleaner visualizations** - Only relevant objects shown  
❌ **Still detects all classes** - Can't speed up model itself

To truly optimize inference for specific classes, you would need to:
1. Fine-tune the model on only those classes
2. Export a smaller model

## Dynamic Filtering

You can make filtering configurable via command-line arguments:

```rust
// Accept classes as command-line args
let args: Vec<String> = env::args().collect();
let allowed_classes: Vec<String> = if args.len() > 3 {
    args[3..].to_vec()  // cargo run ... image.jpg output.jpg airplane car truck
} else {
    vec!["airplane", "car", "truck"]
        .iter()
        .map(|s| s.to_string())
        .collect()
};
```

Usage:
```bash
# Use defaults (person, car, motorcycle, airplane, truck)
cargo run --release --features metal --example detect_video test_data/video.mp4

# Specify classes
cargo run --release --features metal --example detect_video -- --classes 4,5,6 test_data/video.mp4

# Single class (person only)
cargo run --release --features metal --example detect_video -- --classes 0 test_data/video.mp4
```

## Disable Filtering

To show all detected classes, comment out the filter:

```rust
// // Filter detections to only include specific classes
// let allowed_classes = vec!["airplane", "car", "truck"];
// let detections: Vec<_> = detections
//     .into_iter()
//     .filter(|det| {
//         let class_name = det.class.name();
//         allowed_classes.contains(&class_name.as_str())
//     })
//     .collect();
```

Or use an empty filter (allow all):
```rust
let allowed_classes: Vec<&str> = vec![];  // Empty = allow all

let detections: Vec<_> = detections
    .into_iter()
    .filter(|det| {
        if allowed_classes.is_empty() {
            return true;  // Allow all if no filter specified
        }
        let class_name = det.class.name();
        allowed_classes.contains(&class_name.as_str())
    })
    .collect();
```

## Case Sensitivity

Class names are **case-sensitive**:

✅ Correct: `"airplane"`, `"car"`, `"truck"`  
❌ Wrong: `"Airplane"`, `"CAR"`, `"Truck"`

COCO class names are always lowercase.

## Troubleshooting

### No detections after filtering

**Problem**: Detection works but filtering removes everything
```
Total detections: 10
Filtered out: 10 (other classes)
Matching classes: 0
```

**Solution**: Check class names match exactly:
```rust
// Debug: Print all detected class names first
for det in &detections {
    println!("Detected: {}", det.class.name());
}
```

Then update your filter to match the actual class names.

### Wrong class names

**Problem**: Using wrong class names for your model

**Solution**: 
- COCO models: Use COCO class names (see CLASS_SUPPORT.md)
- Custom models: Use `class_0`, `class_1`, etc., or actual training class names
- Check model documentation for class list

## Recommendation

For **aerial vehicle detection**, start with:
```rust
let allowed_classes = vec![
    "airplane",
    "car",
    "truck",
    "bus",
];
```

This covers the most common vehicles visible from aerial imagery while filtering out irrelevant objects like people, animals, furniture, etc.

Adjust based on your specific use case! 🎯
