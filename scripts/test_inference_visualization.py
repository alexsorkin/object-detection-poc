#!/usr/bin/env python3
"""
Test inference on the generated T-80 tank image and visualize results
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import json

def test_inference_on_tank_image():
    """Run inference on the test tank image and visualize results"""
    
    print("\n" + "="*70)
    print("TESTING INFERENCE ON T-80 TANK IMAGE")
    print("="*70 + "\n")
    
    # Paths
    model_path = "models/military_target_detector.pt"
    image_path = "inference/test_data/test_tank.jpg"
    output_path = "inference/test_data/test_tank_detection.jpg"
    
    # Check files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please copy the model:")
        print("  cp runs/detect/*/weights/best.pt models/military_target_detector.pt")
        return
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        print("Please generate it:")
        print("  python scripts/generate_test_image.py")
        return
    
    # Load metadata
    metadata_path = Path(image_path).with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("📋 Test Image Metadata:")
        print(f"  Tank Type: {metadata.get('tank_type', 'Unknown')}")
        print(f"  Position: ({metadata['bbox']['x']:.0f}, {metadata['bbox']['y']:.0f})")
        print(f"  Size: {metadata['bbox']['width']:.0f}x{metadata['bbox']['height']:.0f}")
        print(f"  Expected Class: military_vehicle ({metadata['yolo_format']['class']})")
        print()
    
    # Load model
    print(f"🔧 Loading model: {model_path}")
    model = YOLO(model_path)
    print("  ✓ Model loaded\n")
    
    # Run inference
    print(f"🔍 Running inference on: {image_path}")
    results = model.predict(
        source=image_path,
        conf=0.25,  # Lower confidence threshold for testing
        iou=0.45,
        save=False,
        verbose=False
    )
    
    result = results[0]
    
    # Display results
    print(f"\n📊 DETECTION RESULTS:")
    print("="*70)
    
    if len(result.boxes) == 0:
        print("⚠️  No detections found")
        print("\nPossible reasons:")
        print("  - Model needs more training (only trained for 1 epoch)")
        print("  - Confidence threshold too high")
        print("  - Try training with more epochs: python training/train.py --epochs 50")
    else:
        print(f"✅ Detected {len(result.boxes)} object(s):\n")
        
        class_names = model.names
        
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            print(f"  Detection #{i+1}:")
            print(f"    Class: {class_names[cls_id]} (ID: {cls_id})")
            print(f"    Confidence: {conf:.2%}")
            print(f"    Bounding Box: ({xyxy[0]:.0f}, {xyxy[1]:.0f}) to ({xyxy[2]:.0f}, {xyxy[3]:.0f})")
            print(f"    Size: {xyxy[2]-xyxy[0]:.0f}x{xyxy[3]-xyxy[1]:.0f}")
            print()
    
    # Create visualization
    print(f"🎨 Creating visualization...")
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Try to load a better font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
        font_large = font
    
    # Draw ground truth (if available)
    if metadata_path.exists():
        bbox = metadata['bbox']
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = x1 + bbox['width']
        y2 = y1 + bbox['height']
        
        # Draw ground truth in blue
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
        draw.text((x1, y1 - 25), "Ground Truth: T-80", fill=(0, 0, 255), font=font)
    
    # Draw predictions
    if len(result.boxes) > 0:
        class_names = model.names
        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
        
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([xyxy[0], xyxy[1], xyxy[2], xyxy[3]], 
                          outline=color, width=3)
            
            # Draw label
            label = f"{class_names[cls_id]}: {conf:.2%}"
            draw.text((xyxy[0], xyxy[1] - 25), label, fill=color, font=font)
    
    # Add title
    draw.text((10, 10), "Tank Detection Test - T-80", fill=(255, 255, 255), font=font_large)
    
    # Save
    img.save(output_path, quality=95)
    print(f"  ✓ Saved to: {output_path}")
    
    # Open the result
    print(f"\n📸 Opening result image...")
    import subprocess
    try:
        subprocess.run(["open", output_path], check=True)
    except:
        print(f"  Please open manually: {output_path}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    # Summary
    print("\n💡 Summary:")
    if len(result.boxes) > 0:
        print(f"  ✅ Successfully detected {len(result.boxes)} military vehicle(s)")
        print(f"  Confidence: {float(result.boxes[0].conf[0]):.2%}")
    else:
        print("  ⚠️  No detections - model needs more training")
        print("  Recommendation: Train for 50+ epochs for better accuracy")
    
    print("\n📚 For Rust inference:")
    print("  1. Fix ONNX Runtime integration (see inference/RUST_INFERENCE_GUIDE.md)")
    print("  2. Run: cd inference && cargo run --example real_tank_detection")
    print()

if __name__ == "__main__":
    test_inference_on_tank_image()
