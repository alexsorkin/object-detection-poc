#!/bin/bash

echo "============================================="
echo "ENVIRONMENT CHECK - Tank Detection System"
echo "============================================="
echo ""

# Check virtual environments
echo "1. Checking for virtual environments..."
if [ -d "venv" ]; then
    echo "   ✓ venv/ exists"
    VENV_DIR="venv"
elif [ -d ".venv" ]; then
    echo "   ✓ .venv/ exists"
    VENV_DIR=".venv"
else
    echo "   ✗ No virtual environment found"
    echo ""
    echo "   Create one with:"
    echo "     python3 -m venv venv"
    echo "     source venv/bin/activate"
    echo "     pip install -r requirements-py311.txt"
    exit 1
fi

# Activate and check packages
echo ""
echo "2. Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "   Python: $(which python)"
echo "   Python version: $(python --version)"
echo ""

echo "3. Checking required packages..."
python -c "
import sys

packages = [
    ('cv2', 'opencv-python'),
    ('PIL', 'pillow'),
    ('numpy', 'numpy'),
    ('ultralytics', 'ultralytics'),
    ('torch', 'torch')
]

missing = []
for import_name, package_name in packages:
    try:
        __import__(import_name)
        print(f'   ✓ {package_name}')
    except ImportError:
        print(f'   ✗ {package_name} - MISSING')
        missing.append(package_name)

if missing:
    print(f'\n   Install missing packages:')
    print(f'     pip install -r requirements-py311.txt')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "4. Checking scripts..."
    
    # Check if scripts exist and are executable
    scripts=("workflow.py" "quickstart.sh" "scripts/download_tank_images.py" "scripts/generate_battle_scenes.py" "tests/test_tank_detection.py")
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            if [ -x "$script" ]; then
                echo "   ✓ $script (executable)"
            else
                echo "   ⚠ $script (not executable - run: chmod +x $script)"
            fi
        else
            echo "   ✗ $script (missing)"
        fi
    done
    
    echo ""
    echo "5. Checking directory structure..."
    
    dirs=("data" "models" "scripts" "tests" "training" "inference")
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo "   ✓ $dir/"
        else
            echo "   ⚠ $dir/ (missing - will be created when needed)"
        fi
    done
    
    echo ""
    echo "============================================="
    echo "✓ ENVIRONMENT OK - Ready to use!"
    echo "============================================="
    echo ""
    echo "Next steps:"
    echo "  • Run quick test: ./quickstart.sh"
    echo "  • Generate data: python scripts/generate_battle_scenes.py"
    echo "  • Train model: python training/train.py --epochs 50"
    echo "  • See TANK_DETECTION_WORKFLOW.md for full guide"
    echo ""
else
    echo ""
    echo "============================================="
    echo "✗ ENVIRONMENT CHECK FAILED"
    echo "============================================="
    echo ""
    echo "Please install missing dependencies:"
    echo "  pip install -r requirements-py311.txt"
    echo ""
    exit 1
fi
