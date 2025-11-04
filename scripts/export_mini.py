#!/usr/bin/env python3
"""
Export YOLOv8s (Mini - Fast) model to ONNX format
43MB model with good balance of speed and accuracy
"""

from ultralytics import YOLO

def export_mini_model():
    print("📦 Exporting YOLOv8s (Mini - Fast) Model")
    print("=" * 50)
    
    model = YOLO("yolov8s.pt")
    
    print("\n📤 Exporting to ONNX format...")
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    
    print("\n✅ Export complete!")
    print("📋 Next steps:")
    print("  1. cp yolov8s.onnx ../models/military_target_detector_mini.onnx")
    print("  2. cargo run --release --features metal --example detect_mini")
    print("\n📊 Expected performance: ~180ms (5.6 FPS)")

if __name__ == "__main__":
    export_mini_model()
