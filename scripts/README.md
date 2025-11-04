# Scripts Directory

Utility scripts for model export, testing, and dataset generation.

## Export Scripts

### Export Mini Model (YOLOv8s - Recommended)
```bash
cd scripts
source ../venv/bin/activate
python export_mini.py
```
Exports 43MB fast model (~180ms, 5.6 FPS)

### Export Full Model (YOLOv8m - Accurate)
```bash
cd scripts
source ../venv/bin/activate
python export_full.py
```
Exports 99MB accurate model (~496ms, 2.0 FPS)

## Utility Scripts

- **setup.sh** - Initial environment setup
- **check_env.sh** - Verify Python environment
- **benchmark_gpu.py** - GPU performance testing
- **download_tank_images.py** - Download training dataset
- **generate_battle_scenes.py** - Generate synthetic training data
- **generate_test_image.py** - Create test images
- **test_inference_visualization.py** - Visualize model predictions
- **rust_inference_bridge.py** - Python-Rust inference bridge

## Quick Start

1. **Setup environment:**
   ```bash
   ./setup.sh
   ```

2. **Export models:**
   ```bash
   source venv/bin/activate
   python export_mini.py    # Fast model
   python export_full.py     # Accurate model
   ```

3. **Test in Rust:**
   ```bash
   cd ../inference
   cargo run --release --features metal --example detect_mini
   ```
