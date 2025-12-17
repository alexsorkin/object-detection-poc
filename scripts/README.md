# Scripts Directory

Utility scripts for model export, testing, and dataset generation.

## Export Scripts

Use the export scripts in `training/` directory to export trained RT-DETR models to ONNX format for deployment.

## Utility Scripts

- **setup.sh** - Initial environment setup
- **check_env.sh** - Verify Python environment
- **download_tank_images.py** - Download training dataset
- **generate_battle_scenes.py** - Generate synthetic training data
- **generate_test_image.py** - Create test images
- **create_test_image.py** - Create normalized test images
- **normalize_test_image.py** - Normalize images to model input size

## Quick Start

1. **Setup environment:**
   ```bash
   ./setup.sh
   ```

2. **Train and export models:**
   ```bash
   cd training
   python train.py --config config.yaml
   python export.py --model runs/detect/train/weights/best.pt
   ```

3. **Test in Rust:**
   ```bash
   cd ../inference
   cargo run --release --features metal --example detect_video test_data/airport.mp4
   ```
