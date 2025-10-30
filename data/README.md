# Military Target Detection Dataset

This directory contains the training, validation, and test data for the military target detection model.

## Dataset Structure

```
data/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images  
│   └── test/           # Test images
├── labels/
│   ├── train/          # Training annotations (YOLO format)
│   ├── val/            # Validation annotations
│   └── test/           # Test annotations
└── dataset.yaml        # Dataset configuration
```

## Annotation Format

This project uses YOLO format annotations. Each image has a corresponding `.txt` file with the same name containing bounding box annotations.

### YOLO Format
Each line in the annotation file represents one object:
```
class_id center_x center_y width height
```

Where:
- `class_id`: Integer class identifier (0-3)
- `center_x, center_y`: Normalized center coordinates (0.0-1.0)
- `width, height`: Normalized width and height (0.0-1.0)

### Class Mapping
```
0: armed_personnel     # Soldiers with rifles/weapons
1: rocket_launcher     # RPGs, anti-tank weapons  
2: military_vehicle    # Tanks, armored vehicles, military trucks
3: heavy_weapon        # Mounted guns, artillery pieces
```

## Data Collection Guidelines

### Armed Personnel (Class 0)
- Soldiers carrying rifles, assault weapons
- Personnel with visible weapons
- Combat-ready military personnel
- Various poses, uniforms, environments

### Rocket Launchers (Class 1) 
- RPG-7, RPG-29, anti-tank weapons
- Shoulder-fired missiles
- Portable launcher systems
- Different angles and lighting conditions

### Military Vehicles (Class 2)
- Main battle tanks (T-72, M1 Abrams, etc.)
- Armored personnel carriers (APCs)
- Infantry fighting vehicles (IFVs) 
- Military trucks and support vehicles
- Various camouflage patterns

### Heavy Weapons (Class 3)
- Mounted machine guns
- Artillery pieces
- Anti-aircraft guns
- Crew-served weapons
- Stationary weapon systems

## Data Augmentation

The training pipeline includes automatic augmentation:
- Horizontal flips
- HSV color space variations
- Random scaling and translation
- Mosaic augmentation for small object detection
- Perspective transformations

## Dataset Statistics

Target dataset size:
- Training: 10,000+ images
- Validation: 2,000+ images  
- Test: 1,000+ images

Recommended annotation quality:
- Minimum 50 annotations per class
- Balanced class distribution
- High-quality bounding boxes
- Diverse environments and conditions

## Adding New Data

1. Place images in appropriate subdirectory (`train/`, `val/`, or `test/`)
2. Create corresponding YOLO format annotations in `labels/` subdirectory
3. Ensure consistent file naming (image.jpg -> image.txt)
4. Verify class IDs match the defined mapping
5. Run training script to validate dataset integrity

## Privacy and Ethics

⚠️ **Important**: This dataset is intended for defensive and research purposes only. 
- Ensure all data collection complies with local laws and regulations
- Respect privacy rights and obtain necessary permissions
- Use only publicly available or properly licensed imagery
- Do not include identifying information of individuals