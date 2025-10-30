#!/bin/bash

# Library Upgrade Script
# Updates all Python and Rust dependencies to latest compatible versions

set -e

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

# Upgrade Python dependencies
upgrade_python_deps() {
    print_status "Upgrading Python dependencies..."
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
        print_status "Activated virtual environment"
    else
        print_warning "No virtual environment found. Create one with: python -m venv venv"
    fi
    
    # Upgrade pip first
    pip install --upgrade pip setuptools wheel
    
    # Upgrade all packages from requirements.txt
    if [ -f "requirements.txt" ]; then
        print_status "Upgrading packages from requirements.txt..."
        pip install --upgrade -r requirements.txt
        print_success "Python packages upgraded"
    else
        print_error "requirements.txt not found"
        return 1
    fi
    
    # Install/upgrade additional development tools
    print_status "Installing additional development tools..."
    pip install --upgrade \
        jupyter \
        jupyterlab \
        notebook \
        ipywidgets \
        pre-commit \
        bandit \
        safety
    
    # Show installed versions
    print_status "Current package versions:"
    pip list | grep -E "(torch|ultralytics|opencv|numpy|matplotlib)"
    
    print_success "Python dependencies upgrade complete"
}

# Upgrade Rust dependencies
upgrade_rust_deps() {
    print_status "Upgrading Rust dependencies..."
    
    cd inference
    
    # Update Rust toolchain
    rustup update
    
    # Update cargo itself
    rustup component add cargo
    
    # Update all dependencies
    cargo update
    
    # Audit for security vulnerabilities
    if command -v cargo-audit &> /dev/null; then
        cargo audit
    else
        print_status "Installing cargo-audit..."
        cargo install cargo-audit
        cargo audit
    fi
    
    # Check for outdated dependencies
    if command -v cargo-outdated &> /dev/null; then
        cargo outdated
    else
        print_status "Installing cargo-outdated..."
        cargo install cargo-outdated
        cargo outdated
    fi
    
    # Build to ensure everything works
    cargo build --release
    
    cd ..
    print_success "Rust dependencies upgrade complete"
}

# Upgrade system tools
upgrade_system_tools() {
    print_status "Checking system tools..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "macOS detected"
        
        # Check for Homebrew and update if available
        if command -v brew &> /dev/null; then
            print_status "Updating Homebrew packages..."
            brew update
            brew upgrade python@3.11 || brew upgrade python@3.12 || true
            brew upgrade cmake || true
            brew upgrade pkg-config || true
            print_success "Homebrew packages updated"
        else
            print_warning "Homebrew not found. Consider installing for easier dependency management."
        fi
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Linux detected"
        
        # Update system packages (Ubuntu/Debian)
        if command -v apt &> /dev/null; then
            print_warning "Run 'sudo apt update && sudo apt upgrade' to update system packages"
        # Update system packages (CentOS/RHEL)
        elif command -v yum &> /dev/null; then
            print_warning "Run 'sudo yum update' to update system packages"
        fi
    fi
}

# Clean up old files and caches
cleanup_caches() {
    print_status "Cleaning up caches and temporary files..."
    
    # Python cache cleanup
    if [ -d "venv" ]; then
        source venv/bin/activate
        pip cache purge
    fi
    
    # Remove Python cache directories
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Rust cache cleanup
    if [ -d "inference" ]; then
        cd inference
        cargo clean
        cd ..
    fi
    
    # Remove temporary files
    rm -rf .pytest_cache/ 2>/dev/null || true
    rm -rf .mypy_cache/ 2>/dev/null || true
    
    print_success "Cache cleanup complete"
}

# Update model dependencies
update_model_info() {
    print_status "Updating model requirements..."
    
    cat > models/requirements.txt << 'EOF'
# Model Runtime Requirements
# Minimum versions required for model inference

# ONNX Runtime (choose appropriate version)
onnxruntime>=1.16.0        # CPU version
# onnxruntime-gpu>=1.16.0  # GPU version (CUDA)
# onnxruntime-silicon>=1.16.0  # Apple Silicon optimized

# Core dependencies
numpy>=1.26.0
pillow>=10.1.0

# Optional: For advanced preprocessing
opencv-python>=4.9.0
scipy>=1.11.0
EOF

    print_success "Model requirements updated"
}

# Validate installation
validate_installation() {
    print_status "Validating installation..."
    
    # Activate virtual environment
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Test Python imports
    python -c "
import torch
import torchvision
import ultralytics
import cv2
import numpy as np
import matplotlib
print('✅ Core Python packages imported successfully')

# Test PyTorch device detection
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ Apple Metal (MPS) available')
else:
    print('ℹ️  Using CPU (no GPU acceleration)')

print(f'PyTorch version: {torch.__version__}')
print(f'Ultralytics version: {ultralytics.__version__}')
"
    
    if [ $? -eq 0 ]; then
        print_success "Python validation passed"
    else
        print_error "Python validation failed"
        return 1
    fi
    
    # Test Rust compilation
    if [ -d "inference" ]; then
        cd inference
        if cargo check --quiet; then
            print_success "Rust validation passed"
        else
            print_error "Rust validation failed"
            cd ..
            return 1
        fi
        cd ..
    fi
    
    print_success "All validations passed!"
}

# Generate updated requirements files
generate_lockfiles() {
    print_status "Generating dependency lock files..."
    
    # Python requirements with exact versions
    if [ -d "venv" ]; then
        source venv/bin/activate
        pip freeze > requirements-lock.txt
        print_status "Generated requirements-lock.txt with exact versions"
    fi
    
    # Rust Cargo.lock is generated automatically
    if [ -d "inference" ]; then
        cd inference
        cargo build --quiet
        cd ..
        print_status "Updated Cargo.lock"
    fi
    
    print_success "Lock files generated"
}

# Main upgrade function
main() {
    echo
    print_status "🚀 Starting dependency upgrade process..."
    echo
    
    upgrade_system_tools
    echo
    
    upgrade_python_deps
    echo
    
    upgrade_rust_deps
    echo
    
    update_model_info
    echo
    
    cleanup_caches
    echo
    
    validate_installation
    echo
    
    generate_lockfiles
    echo
    
    print_success "🎉 All dependencies upgraded successfully!"
    echo
    
    print_status "Summary of changes:"
    print_status "- Updated Python packages to latest versions"
    print_status "- Updated Rust dependencies"
    print_status "- Cleaned up caches and temporary files"
    print_status "- Generated lock files for reproducible builds"
    echo
    
    print_status "Next steps:"
    print_status "1. Test your training: python test_mac_gpu.py"
    print_status "2. Run training: python training/train.py --config config_mac.yaml"
    print_status "3. Commit the updated requirements files"
    echo
    
    print_warning "Note: If you encounter issues, check the generated lock files"
    print_warning "for exact versions that work on your system."
}

# Run main function
main "$@"