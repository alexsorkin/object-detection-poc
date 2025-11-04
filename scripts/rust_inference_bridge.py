#!/usr/bin/env python3
"""
Rust-Python Inference Bridge
This script provides inference capabilities for the Rust library by calling Python/YOLOv8
"""

import sys
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_inference(image_path: str, model_path: str, conf_threshold: float = 0.25) -> dict:
    """
    Run inference on an image and return JSON results
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model (.pt or .onnx)
        conf_threshold: Confidence threshold
    
    Returns:
        Dictionary with detection results
    """
    # Check files exist
    if not Path(image_path).exists():
        return {"error": f"Image not found: {image_path}", "detections": []}
    
    if not Path(model_path).exists():
        return {"error": f"Model not found: {model_path}", "detections": []}
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}", "detections": []}
    
    # Run inference
    try:
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=0.45,
            save=False,
            verbose=False
        )
    except Exception as e:
        return {"error": f"Inference failed: {e}", "detections": []}
    
    # Parse results
    detections = []
    class_names = model.names
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            detection = {
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "confidence": conf,
                "bbox": {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "center_x": float((xyxy[0] + xyxy[2]) / 2),
                    "center_y": float((xyxy[1] + xyxy[3]) / 2),
                    "width": float(xyxy[2] - xyxy[0]),
                    "height": float(xyxy[3] - xyxy[1]),
                }
            }
            detections.append(detection)
    
    return {
        "success": True,
        "detections": detections,
        "count": len(detections),
        "image_path": image_path,
        "model_path": model_path,
    }

def main():
    parser = argparse.ArgumentParser(description='Rust-Python Inference Bridge')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', required=True, help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON')
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(args.image, args.model, args.conf)
    
    # Output JSON
    if args.pretty:
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps(results))

if __name__ == '__main__':
    main()
