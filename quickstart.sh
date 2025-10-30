#!/bin/bash

# Quick start script for tank detection model training and testing
# This runs a minimal workflow for testing the complete pipeline

echo "=========================================="
echo "TANK DETECTION - QUICK START"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Generate 50 synthetic battle scenes"
echo "  2. Train YOLOv8 model for 10 epochs"
echo "  3. Test on 10 sample images"
echo ""
echo "Expected time: ~5-10 minutes on Mac M1/M2"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo ""
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run quick workflow
echo ""
echo "Starting quick workflow..."
python3 workflow.py \
  --generate \
  --train \
  --test \
  --epochs 10 \
  --batch-size 8 \
  --max-test-images 10

echo ""
echo "=========================================="
echo "QUICK START COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  • Review results in data/synthetic_scenes/test_results.json"
echo "  • Run full training: python3 workflow.py --train --epochs 50"
echo "  • Test live video: python3 tests/test_tank_detection.py --mode video"
echo "  • See TANK_DETECTION_WORKFLOW.md for detailed documentation"
echo ""
