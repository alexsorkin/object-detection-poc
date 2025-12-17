# Object Class Support

## Overview

The detection system supports object classes from standard datasets:
- **COCO** (80 classes) - Most common objects  
- **Objects365** (365 classes) - Comprehensive object detection dataset
- **Custom trained models** - Any class definitions

## Implementation

### TargetClass Enum

The `TargetClass` enum has been simplified to a single variant:

```rust
pub enum TargetClass {
    Class(u32),  // Universal class with ID
}
```

### Automatic Class Name Mapping

The system automatically resolves class names based on the class ID:

1. **COCO Classes** (IDs 0-79): Standard COCO dataset classes
2. **Objects365 Classes** (IDs 0-364): Extended object detection classes
3. **Fallback**: For any unknown class ID, returns `class_N` format

### Supported COCO Classes (80 classes)

```
0: person          20: elephant       40: wine glass     60: dining table
1: bicycle         21: bear           41: cup            61: toilet
2: car             22: zebra          42: fork           62: tv
3: motorcycle      23: giraffe        43: knife          63: laptop
4: airplane        24: backpack       44: spoon          64: mouse
5: bus             25: umbrella       45: bowl           65: remote
6: train           26: handbag        46: banana         66: keyboard
7: truck           27: tie            47: apple          67: cell phone
8: boat            28: suitcase       48: sandwich       68: microwave
9: traffic light   29: frisbee        49: orange         69: oven
10: fire hydrant   30: skis           50: broccoli       70: toaster
11: stop sign      31: snowboard      51: carrot         71: sink
12: parking meter  32: sports ball    52: hot dog        72: refrigerator
13: bench          33: kite           53: pizza          73: book
14: bird           34: baseball bat   54: donut          74: clock
15: cat            35: baseball glove 55: cake           75: vase
16: dog            36: skateboard     56: chair          76: scissors
17: horse          37: surfboard      57: couch          77: teddy bear
18: sheep          38: tennis racket  58: potted plant   78: hair drier
19: cow            39: bottle         59: bed            79: toothbrush
```

### Supported Objects365 Classes (365 classes)

Objects365 includes all common objects, furniture, vehicles, food items, electronics, and many more. See the full list in `src/types.rs`.

Notable categories:
- **Vehicles**: car, truck, bus, motorcycle, bicycle, van, suv, pickup truck, ambulance, fire truck, etc.
- **People & Clothing**: person, hat, glasses, shoes, gloves, backpack, handbag, etc.
- **Furniture**: chair, desk, bed, couch, table, shelf, cabinet, etc.
- **Electronics**: tv, laptop, phone, camera, keyboard, mouse, etc.
- **Food**: fruits, vegetables, prepared dishes, beverages, etc.
- **Animals**: dog, cat, bird, horse, cow, fish, elephant, etc.
- **Tools & Equipment**: various tools, sports equipment, musical instruments, etc.

## Color Generation

Colors are automatically generated using a deterministic algorithm based on class ID:
- Uses golden angle (137°) for optimal color distribution
- HSV to RGB conversion for consistent, visually distinct colors
- Ensures different classes have different colors

## Usage

### Detecting with RT-DETR

```rust
use military_target_detector::{DetectorConfig, RTDETRDetector};

let config = DetectorConfig {
    model_path: "models/rtdetr_v2_r50vd_fp32.onnx".to_string(),
    confidence_threshold: 0.5,
    ..Default::default()
};

let detector = RTDETRDetector::new(config)?;
let detections = detector.detect(&image)?;

for detection in detections {
    println!("Class: {} (ID: {})", 
        detection.class.name(), 
        detection.class as u32
    );
}
```

### Example Models

1. **COCO-trained RT-DETR**: Detects 80 common objects
   - Models: rtdetr_v2_r18vd, rtdetr_v2_r50vd, rtdetr_v2_r101vd
   - Classes: person, car, bicycle, dog, cat, etc.

2. **Objects365-trained models**: Detects 365 object classes
   - More comprehensive object detection
   - Better coverage of uncommon objects

3. **Custom Models**: Any model with class IDs 0-N
   - Unknown classes automatically labeled as `class_N`

## Performance

The system handles all class IDs efficiently:
- O(1) lookup for COCO classes (0-79)
- O(1) lookup for Objects365 classes (0-364)
- Fallback formatting for any other class ID

## Backwards Compatibility

Military target detection classes (0-3) still work:
```rust
let military_classes = TargetClass::all(); // Returns IDs 0-3
```

Class names for military models:
- 0: Armed Personnel
- 1: Rocket Launcher
- 2: Military Vehicle
- 3: Heavy Weapon

## Migration Notes

If you have code using the old enum variants:

**Before:**
```rust
match class {
    TargetClass::ArmedPersonnel => { /* ... */ }
    TargetClass::Car => { /* ... */ }
    _ => { /* ... */ }
}
```

**After:**
```rust
match class {
    TargetClass::Class(0) => { /* Armed Personnel */ }
    TargetClass::Class(2) => { /* Car */ }
    TargetClass::Class(id) => {
        let name = class.name();
        // Use name string instead
    }
}
```

**Recommended approach:**
```rust
// Use the name() method for display/logic
let class_name = detection.class_name;
match class_name.as_str() {
    "person" => { /* ... */ }
    "car" => { /* ... */ }
    _ => { /* ... */ }
}
```

## References

- COCO Dataset: https://cocodataset.org/
- Objects365 Dataset: https://www.objects365.org/
- RT-DETR: https://github.com/lyuwenyu/RT-DETR
