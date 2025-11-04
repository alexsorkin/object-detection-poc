#!/bin/bash

# Military Target Detection - Setup Script
# This script sets up the complete development environment

set -e

echo "🎯 Military Target Detection - Setup Script"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        print_warning "Rust is not installed. Installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is required but not installed"
        exit 1
    fi
    
    print_success "System requirements check complete"
}

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Installed Python dependencies"
    fi
    
    # Install PyTorch with Mac optimization
    setup_pytorch_for_mac
    
    print_success "Python environment setup complete"
}

# Setup PyTorch optimized for Mac
setup_pytorch_for_mac() {
    print_status "Setting up PyTorch for Mac..."
    
    # Detect Mac architecture
    if [[ $(uname -s) == "Darwin" ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            print_status "Apple Silicon Mac detected - installing optimized PyTorch"
            # Install latest stable with MPS support
            pip install --upgrade torch torchvision torchaudio
            
            # Verify MPS availability
            python -c "
import torch
print('PyTorch version:', torch.__version__)
if hasattr(torch.backends, 'mps'):
    print('MPS available:', torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        print('✅ Mac GPU acceleration ready!')
    else:
        print('⚠️  MPS not available - check macOS version (need 12.3+)')
else:
    print('⚠️  MPS backend not found - using CPU')
" 2>/dev/null || print_warning "Could not verify MPS support"
            
        else
            print_status "Intel Mac detected - installing standard PyTorch"
            pip install --upgrade torch torchvision torchaudio
        fi
    else
        print_status "Non-Mac system - installing PyTorch"
        # Let pip choose the best version for the system
        pip install --upgrade torch torchvision torchaudio
    fi
}

# Setup Rust environment
setup_rust() {
    print_status "Setting up Rust environment..."
    
    cd inference
    
    # Add required targets for cross-compilation
    rustup target add x86_64-pc-windows-gnu
    rustup target add aarch64-apple-darwin
    rustup target add aarch64-linux-android
    
    # Install cargo extensions
    cargo install cargo-edit
    cargo install cargo-audit
    
    # Build library
    cargo build --release
    
    cd ..
    print_success "Rust environment setup complete"
}

# Download pre-trained models
download_models() {
    print_status "Setting up models directory..."
    
    mkdir -p models
    
    # Create placeholder model files
    cat > models/README.md << 'EOF'
# Models Directory

This directory should contain your trained ONNX models.

## Expected Files:
- `military_targets.onnx` - Main detection model
- `model_info.json` - Model metadata

## Training Your Own Model:
1. Prepare dataset in YOLO format
2. Run training: `python training/train.py`
3. Export model: `python training/export.py`

## Pre-trained Models:
Due to the sensitive nature of military target detection, pre-trained models
are not distributed. You must train your own models using appropriate,
legally obtained datasets.
EOF

    print_warning "No pre-trained models available. You need to train your own model."
    print_status "See data/README.md for dataset preparation instructions"
}

# Setup development tools
setup_dev_tools() {
    print_status "Setting up development tools..."
    
    # Python development tools
    source venv/bin/activate
    pip install black flake8 pytest jupyter notebook
    
    # Rust development tools
    rustup component add clippy rustfmt
    
    # Git hooks
    if [ ! -f ".git/hooks/pre-commit" ]; then
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run code formatting and linting before commit

echo "Running pre-commit checks..."

# Python formatting
if command -v black &> /dev/null; then
    black training/ --check
fi

# Rust formatting
if command -v cargo &> /dev/null; then
    cd inference && cargo fmt -- --check && cd ..
fi

# Rust linting
if command -v cargo &> /dev/null; then
    cd inference && cargo clippy -- -D warnings && cd ..
fi

echo "Pre-commit checks passed"
EOF
        chmod +x .git/hooks/pre-commit
        print_success "Git pre-commit hooks installed"
    fi
}

# Create example scripts
create_examples() {
    print_status "Creating example scripts..."
    
    # Training example
    cat > train_model.sh << 'EOF'
#!/bin/bash
# Example training script

echo "Starting model training..."

# Activate Python environment
source venv/bin/activate

# Run training
cd training
python train.py --config config.yaml

# Export model
python export.py ../models/military_targets_best.pt --output-dir ../models --formats onnx

echo "Training complete. Model exported to models/military_targets.onnx"
EOF
    chmod +x train_model.sh
    
    # Inference example
    cat > test_detection.sh << 'EOF'
#!/bin/bash
# Example detection script

echo "Testing detection on sample image..."

# Build Rust library
cd inference
cargo build --release

# Run detection
./target/release/examples/basic_detection \
    --model ../models/military_targets.onnx \
    --image ../data/sample.jpg \
    --output detection_result.jpg

echo "Detection complete. Result saved as detection_result.jpg"
EOF
    chmod +x test_detection.sh
    
    print_success "Example scripts created"
}

# Setup project structure
setup_structure() {
    print_status "Setting up project structure..."
    
    # Create missing directories
    mkdir -p data/images/{train,val,test}
    mkdir -p data/labels/{train,val,test}
    mkdir -p models
    mkdir -p logs
    mkdir -p output
    
    # Create .gitignore
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Rust
target/
**/*.rs.bk
Cargo.lock

# Models and data
*.onnx
*.pt
*.pth
*.bin
data/images/
data/labels/
!data/images/.gitkeep
!data/labels/.gitkeep

# Logs
logs/
*.log

# Output
output/
runs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# System
.DS_Store
Thumbs.db

# Unity
Library/
Temp/
obj/
Build/
Builds/
Assets/AssetStoreTools*
*.csproj
*.unityproj
*.sln
*.suo
*.tmp
*.user
*.userprefs
*.pidb
*.booproj
*.svd
*.pdb
*.opendb
*.VC.db
EOF
    fi
    
    # Create .gitkeep files
    touch data/images/.gitkeep
    touch data/labels/.gitkeep
    touch models/.gitkeep
    
    print_success "Project structure setup complete"
}

# Main setup function
main() {
    echo
    print_status "Starting setup process..."
    echo
    
    check_requirements
    echo
    
    setup_structure
    echo
    
    setup_python
    echo
    
    setup_rust
    echo
    
    download_models
    echo
    
    setup_dev_tools
    echo
    
    create_examples
    echo
    
    print_success "Setup complete! 🎉"
    echo
    
    echo "Next steps:"
    echo "1. Prepare your training dataset in data/ directory"
    echo "2. Run ./train_model.sh to train a model"
    echo "3. Use ./test_detection.sh to test detection"
    echo "4. See examples/README.md for more usage examples"
    echo
    
    print_warning "Remember: This system is for defensive research purposes only."
    print_warning "Ensure compliance with local laws and ethical guidelines."
}

# Run main function
main "$@"