#!/usr/bin/env python3
"""
Export YOLOv8m (Full - Accurate) model to ONNX format
99MB model with highest accuracy
"""

from ultralytics import YOLO

def export_full_model():
    print("📦 Exporting YOLOv8m (Full - Accurate) Model")
    print("=" * 50)
    
    model = YOLO("yolov8m.pt")
    
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
    print("  1. cp yolov8m.onnx ../models/military_target_detector.onnx")
    print("  2. cargo run --release --features metal --example detect_full")
    print("\n📊 Expected performance: ~496ms (2.0 FPS)")

if __name__ == "__main__":
    export_full_model()
