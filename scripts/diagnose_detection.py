#!/usr/bin/env python3
"""
Diagnose why airplane detection might be failing
Tests different confidence thresholds and analyzes the image
"""

import sys
sys.path.append('..')

from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

def test_detection(image_path, model_path, confidence_thresholds=[0.01, 0.05, 0.10, 0.25, 0.5]):
    """
    Test detection with various confidence thresholds
    """
    print(f"🔍 Diagnosing Detection Issues")
    print(f"=" * 60)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print()
    
    # Load image
    try:
        img = Image.open(image_path)
        print(f"✓ Image loaded: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return
    
    # Load model
    try:
        if model_path.endswith('.pt'):
            model = YOLO(model_path)
            print(f"✓ PyTorch model loaded")
        else:
            print(f"✗ Can only diagnose .pt models with this script")
            print(f"  For ONNX models, use the Rust detector")
            return
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print()
    print(f"Testing different confidence thresholds:")
    print("-" * 60)
    
    best_conf = None
    best_count = 0
    
    for conf in confidence_thresholds:
        results = model(img, conf=conf, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            num_detections = len(boxes)
            
            print(f"Confidence {conf:5.2f} ({conf*100:5.1f}%): {num_detections:3d} detections")
            
            if num_detections > best_count:
                best_count = num_detections
                best_conf = conf
            
            # Show details for first few detections
            for i, box in enumerate(boxes[:5]):  # Show first 5
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                print(f"    #{i+1}: class={cls_id}, conf={confidence:.3f}")
        else:
            print(f"Confidence {conf:5.2f} ({conf*100:5.1f}%):   0 detections")
    
    print()
    if best_count > 0:
        print(f"✓ Best threshold: {best_conf} ({best_conf*100:.1f}%) with {best_count} detections")
        print()
        print(f"Recommendation: Use confidence_threshold: {best_conf}")
    else:
        print(f"✗ NO DETECTIONS FOUND at any threshold!")
        print()
        print("Possible issues:")
        print("1. Model not trained on objects in this image")
        print("2. Image quality/resolution too low")
        print("3. Objects too small or occluded")
        print("4. Wrong model for this task")
        print()
        print("Try:")
        print("- Use a different model (e.g., COCO-trained YOLOv8)")
        print("- Check if objects are visible in the image")
        print("- Verify model is trained on relevant classes")

def main():
    # Test with airplane image
    image_path = "inference/test_data/yolo_airport.jpg"
    
    # Try to find a .pt model
    import os
    
    pt_models = [
        "models/yolo_aireal_detector.pt",
        "pretrained_models/yolov8s.pt",
        "pretrained_models/yolov8m.pt",
    ]
    
    for model_path in pt_models:
        if os.path.exists(model_path):
            print(f"Testing with model: {model_path}\n")
            test_detection(image_path, model_path)
            print("\n" + "=" * 60 + "\n")
            break
    else:
        print("No .pt models found for diagnosis")
        print("Available models:")
        for model in pt_models:
            print(f"  - {model} {'✓' if os.path.exists(model) else '✗'}")

if __name__ == '__main__':
    main()
