#!/usr/bin/env python3
"""
Mac GPU Detection and Performance Test
Tests Apple Silicon GPU capabilities for military target detection training
"""

import torch
import time
import sys
import platform
import subprocess

def get_system_info():
    """Get Mac system information"""
    print("🖥️  System Information")
    print("=" * 50)
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Get macOS version
    try:
        macos_version = subprocess.check_output(['sw_vers', '-productVersion'], 
                                              text=True).strip()
        print(f"macOS Version: {macos_version}")
    except:
        print("macOS Version: Unknown")
    
    # Get memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.total // (1024**3)}GB total")
        print(f"Memory Available: {memory.available // (1024**3)}GB")
    except ImportError:
        print("Memory info: Install psutil for details")
    
    print()

def test_pytorch_installation():
    """Test PyTorch installation and MPS availability"""
    print("🔥 PyTorch Information")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    
    # Check MPS availability
    if hasattr(torch.backends, 'mps'):
        print(f"MPS Available: {torch.backends.mps.is_available()}")
        if hasattr(torch.backends.mps, 'is_built'):
            print(f"MPS Built: {torch.backends.mps.is_built()}")
    else:
        print("MPS Backend: Not found (need PyTorch 1.12+)")
    
    # Check CUDA (for comparison)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    
    print()

def benchmark_devices():
    """Benchmark different device performance"""
    print("⚡ Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    size = (1000, 1000)
    iterations = 100
    
    # Available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    
    results = {}
    
    for device in devices:
        print(f"\nTesting {device.upper()}...")
        
        try:
            # Create tensors
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)
            
            # Warmup
            for _ in range(10):
                _ = torch.mm(a, b)
            
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            
            for _ in range(iterations):
                c = torch.mm(a, b)
            
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / iterations * 1000  # ms
            
            results[device] = avg_time
            print(f"  Average time: {avg_time:.2f}ms")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[device] = float('inf')
    
    # Show speedup comparison
    if 'cpu' in results and results['cpu'] != float('inf'):
        cpu_time = results['cpu']
        print(f"\n📊 Speedup vs CPU:")
        for device, time_ms in results.items():
            if device != 'cpu' and time_ms != float('inf'):
                speedup = cpu_time / time_ms
                print(f"  {device.upper()}: {speedup:.1f}x faster")
    
    print()

def test_ml_workload():
    """Test a more realistic ML workload"""
    print("🧠 ML Workload Test")
    print("=" * 50)
    
    try:
        # Test device selection logic
        if torch.backends.mps.is_available():
            device = 'mps'
            print("✅ Using MPS (Mac GPU)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("✅ Using CUDA")
        else:
            device = 'cpu'
            print("⚠️  Using CPU (no GPU available)")
        
        # Simple CNN-like operations
        print(f"\nTesting on {device}...")
        
        # Create sample data similar to YOLO input
        batch_size = 16
        channels = 3
        height = 640
        width = 640
        
        x = torch.randn(batch_size, channels, height, width, device=device)
        
        # Simple convolution (similar to YOLO backbone)
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        
        start_time = time.time()
        
        # Forward pass
        with torch.no_grad():
            y = conv(x)
            y = torch.relu(y)
            y = torch.nn.functional.max_pool2d(y, 2)
        
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # ms
        print(f"  Batch inference time: {inference_time:.2f}ms")
        print(f"  Per-image time: {inference_time/batch_size:.2f}ms")
        print(f"  Estimated FPS: {1000/(inference_time/batch_size):.1f}")
        
        # Memory usage
        if device == 'mps':
            try:
                allocated = torch.mps.current_allocated_memory() / 1e9
                print(f"  MPS memory allocated: {allocated:.2f}GB")
            except:
                pass
        elif device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"  CUDA memory allocated: {allocated:.2f}GB")
        
        print("✅ ML workload test completed successfully")
        
    except Exception as e:
        print(f"❌ ML workload test failed: {e}")
    
    print()

def training_recommendations():
    """Provide training recommendations based on system"""
    print("💡 Training Recommendations")
    print("=" * 50)
    
    # Get memory info
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total // (1024**3)
    except:
        memory_gb = 16  # Default assumption
    
    # Check if MPS is available
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if has_mps:
        print("✅ Mac GPU (MPS) detected!")
        print("\nRecommended configuration:")
        
        if memory_gb >= 32:
            print("  Config file: config_mac.yaml")
            print("  Batch size: 32")
            print("  Workers: 8")
            print("  Image size: 640")
            print("  Model: yolov8s.pt (can handle larger models)")
        elif memory_gb >= 16:
            print("  Config file: config_mac.yaml")
            print("  Batch size: 16")
            print("  Workers: 6")
            print("  Image size: 640")
            print("  Model: yolov8n.pt or yolov8s.pt")
        else:
            print("  Config file: config_mac_8gb.yaml")
            print("  Batch size: 8")
            print("  Workers: 4")
            print("  Image size: 416")
            print("  Model: yolov8n.pt (nano only)")
        
        print("\nTraining command:")
        if memory_gb <= 8:
            print("  python train.py --config config_mac_8gb.yaml")
        else:
            print("  python train.py --config config_mac.yaml")
            
    else:
        print("⚠️  No GPU acceleration available")
        print("Recommendations:")
        print("  - Use CPU training (will be slower)")
        print("  - Consider cloud GPU training for large datasets")
        print("  - Update to macOS 12.3+ for MPS support")
    
    print()

def main():
    """Run all tests and provide recommendations"""
    print("🎯 Mac GPU Training Compatibility Test")
    print("=" * 70)
    print()
    
    get_system_info()
    test_pytorch_installation()
    benchmark_devices()
    test_ml_workload()
    training_recommendations()
    
    print("🏁 Test completed!")
    print("\nNext steps:")
    print("1. Use recommended configuration file")
    print("2. Run: python training/train.py --config <recommended_config>")
    print("3. Monitor system memory and temperature during training")
    print("4. See MAC_GPU_SETUP.md for detailed optimization guide")

if __name__ == "__main__":
    main()