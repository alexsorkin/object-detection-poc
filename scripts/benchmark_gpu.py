#!/usr/bin/env python3
"""
GPU Inference Benchmark
Tests inference speed with Metal GPU acceleration on your AMD Radeon Pro 5500M
"""

import time
from ultralytics import YOLO
import torch

def benchmark_inference(model_path: str, image_path: str, num_runs: int = 5):
    """Benchmark inference with and without GPU"""
    
    print("🚀 GPU Inference Benchmark\n")
    print(f"GPU: {torch.backends.mps.is_available()}")
    print(f"Device: AMD Radeon Pro 5500M (8GB VRAM)\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Test 1: CPU inference
    print("=" * 60)
    print("TEST 1: CPU Inference (Current)")
    print("=" * 60)
    
    model.to('cpu')
    cpu_times = []
    
    for i in range(num_runs):
        start = time.time()
        results = model(image_path, verbose=False)
        elapsed = time.time() - start
        cpu_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.1f}ms")
    
    cpu_avg = sum(cpu_times) / len(cpu_times)
    print(f"\n  Average: {cpu_avg*1000:.1f}ms ({1/cpu_avg:.2f} FPS)")
    
    # Test 2: GPU inference with Metal
    print("\n" + "=" * 60)
    print("TEST 2: GPU Inference (Metal/AMD Radeon)")
    print("=" * 60)
    
    try:
        model.to('mps')  # Metal Performance Shaders (Apple's GPU API)
        gpu_times = []
        
        # Warmup
        print("  Warming up GPU...")
        for _ in range(2):
            _ = model(image_path, verbose=False)
        
        print("  Running benchmark...")
        for i in range(num_runs):
            start = time.time()
            results = model(image_path, verbose=False)
            elapsed = time.time() - start
            gpu_times.append(elapsed)
            print(f"  Run {i+1}: {elapsed*1000:.1f}ms")
        
        gpu_avg = sum(gpu_times) / len(gpu_times)
        print(f"\n  Average: {gpu_avg*1000:.1f}ms ({1/gpu_avg:.2f} FPS)")
        
        # Comparison
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"  CPU:     {cpu_avg*1000:.1f}ms  ({1/cpu_avg:.2f} FPS)")
        print(f"  GPU:     {gpu_avg*1000:.1f}ms  ({1/gpu_avg:.2f} FPS)")
        print(f"  Speedup: {cpu_avg/gpu_avg:.1f}x faster ⚡")
        
        # Get detection info
        if len(results[0].boxes) > 0:
            print(f"\n  ✅ Detected {len(results[0].boxes)} object(s)")
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                print(f"     - Class {cls} with {conf*100:.1f}% confidence")
        
    except Exception as e:
        print(f"\n  ❌ GPU inference failed: {e}")
        print("  (This is expected on older Macs or if PyTorch doesn't support your GPU)")

if __name__ == "__main__":
    MODEL_PATH = "models/military_target_detector.pt"
    IMAGE_PATH = "inference/test_data/test_tank.jpg"
    
    benchmark_inference(MODEL_PATH, IMAGE_PATH, num_runs=5)
    
    print("\n💡 TIP: If GPU is slower, it might be due to:")
    print("   - Data transfer overhead (small images)")
    print("   - Model not optimized for GPU")
    print("   - First-run compilation costs")
    print("\n   GPU shines with larger images or batch processing!")
