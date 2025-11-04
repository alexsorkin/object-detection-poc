#!/usr/bin/env python3
"""
Generate a test image with T-80 tank for Rust inference testing
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_test_image_with_tank(output_path: str = "inference/test_data/test_tank.jpg"):
    """Generate a test battle scene with a T-80 tank"""
    
    print("Generating test image with T-80 tank...")
    
    # Image dimensions
    width, height = 640, 640
    
    # Load a background
    backgrounds_dir = Path("data/backgrounds")
    backgrounds = list(backgrounds_dir.glob("*.jpg")) + list(backgrounds_dir.glob("*.png"))
    
    if backgrounds:
        background = Image.open(random.choice(backgrounds)).convert("RGB")
        background = background.resize((width, height))
    else:
        # Create a simple battlefield background
        background = Image.new("RGB", (width, height), color=(139, 119, 101))  # Brown/tan
        draw = ImageDraw.Draw(background)
        # Add some ground texture
        for _ in range(100):
            x = random.randint(0, width)
            y = random.randint(height//2, height)
            w = random.randint(5, 20)
            h = random.randint(5, 20)
            color = (random.randint(100, 150), random.randint(90, 130), random.randint(70, 110))
            draw.ellipse([x, y, x+w, y+h], fill=color)
    
    # Load T-80 tank image
    tank_dir = Path("data/raw_images/tanks")
    t80_images = list(tank_dir.glob("T-80*.jpg")) + list(tank_dir.glob("T-80*.png"))
    
    if not t80_images:
        print("❌ No T-80 images found. Downloading...")
        os.system("python scripts/download_tank_images.py --models T-80 --max-per-model 5")
        t80_images = list(tank_dir.glob("T-80*.jpg")) + list(tank_dir.glob("T-80*.png"))
    
    if not t80_images:
        print("❌ Could not find T-80 images. Creating synthetic tank...")
        # Create a simple tank shape
        tank_img = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tank_img)
        # Tank body
        draw.rectangle([20, 40, 180, 80], fill=(80, 80, 60), outline=(50, 50, 30), width=2)
        # Tank turret
        draw.ellipse([60, 20, 140, 60], fill=(70, 70, 50), outline=(50, 50, 30), width=2)
        # Tank barrel
        draw.rectangle([140, 35, 200, 45], fill=(60, 60, 40), outline=(40, 40, 20), width=2)
        # Wheels
        for x in [40, 70, 100, 130, 160]:
            draw.ellipse([x-10, 70, x+10, 90], fill=(40, 40, 40), outline=(20, 20, 20), width=2)
        tank = tank_img
    else:
        # Load a real T-80 image
        tank_path = random.choice(t80_images)
        print(f"Using tank image: {tank_path.name}")
        try:
            tank = Image.open(tank_path).convert("RGBA")
        except Exception as e:
            print(f"Error loading {tank_path}: {e}")
            # Use synthetic fallback
            tank_img = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
            draw = ImageDraw.Draw(tank_img)
            draw.rectangle([20, 40, 180, 80], fill=(80, 80, 60), outline=(50, 50, 30), width=2)
            draw.ellipse([60, 20, 140, 60], fill=(70, 70, 50), outline=(50, 50, 30), width=2)
            draw.rectangle([140, 35, 200, 45], fill=(60, 60, 40), outline=(40, 40, 20), width=2)
            tank = tank_img
    
    # Resize and rotate tank
    tank_width = random.randint(150, 250)
    tank_height = int(tank.height * (tank_width / tank.width))
    tank = tank.resize((tank_width, tank_height), Image.Resampling.LANCZOS)
    
    # Rotate slightly for realism
    rotation = random.randint(-15, 15)
    tank = tank.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    
    # Position tank in the scene
    x = random.randint(width//4, width - tank.width - width//4)
    y = random.randint(height//2, height - tank.height - 50)
    
    # Paste tank onto background
    background.paste(tank, (x, y), tank if tank.mode == 'RGBA' else None)
    
    # Save the image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path, quality=95)
    
    # Generate YOLO label
    label_path = output_path.with_suffix('.txt')
    
    # Calculate bounding box in YOLO format (normalized)
    bbox_x_center = (x + tank.width / 2) / width
    bbox_y_center = (y + tank.height / 2) / height
    bbox_width = tank.width / width
    bbox_height = tank.height / height
    
    # Class 2 is military_vehicle
    with open(label_path, 'w') as f:
        f.write(f"2 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
    
    # Save metadata
    metadata = {
        "image_path": str(output_path),
        "tank_type": "T-80",
        "bbox": {
            "x": x,
            "y": y,
            "width": tank.width,
            "height": tank.height,
            "x_center": x + tank.width / 2,
            "y_center": y + tank.height / 2
        },
        "yolo_format": {
            "class": 2,
            "x_center": bbox_x_center,
            "y_center": bbox_y_center,
            "width": bbox_width,
            "height": bbox_height
        }
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Test image generated successfully!")
    print(f"   Image: {output_path}")
    print(f"   Label: {label_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"\nTank details:")
    print(f"   Type: T-80")
    print(f"   Position: ({x}, {y})")
    print(f"   Size: {tank.width}x{tank.height}")
    print(f"   Rotation: {rotation}°")
    print(f"\nExpected detection:")
    print(f"   Class: military_vehicle (2)")
    print(f"   Bbox (normalized): center=({bbox_x_center:.3f}, {bbox_y_center:.3f}), size=({bbox_width:.3f}x{bbox_height:.3f})")
    
    return output_path, metadata

if __name__ == "__main__":
    generate_test_image_with_tank()
