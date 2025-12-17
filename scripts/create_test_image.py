#!/usr/bin/env python3
"""
Create a normalized test image for aerial detection
Expected size: 640x640 (standard model input size)
"""

import os
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

def create_test_image_from_url(output_path, size=(640, 640)):
    """
    Download a sample aerial image and resize it to model input size
    """
    # Sample aerial images from unsplash (free to use)
    aerial_urls = [
        # Aerial view of city with cars
        "https://images.unsplash.com/photo-1486299267070-83823f5448dd?w=1200",
        # Aerial view of parking lot
        "https://images.unsplash.com/photo-1449157291145-7efd050a4d0e?w=1200",
        # Drone view of vehicles
        "https://images.unsplash.com/photo-1473163928189-364b2c4e1135?w=1200",
    ]
    
    print(f"Downloading sample aerial image...")
    
    for url in aerial_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    else:
        print("Could not download image, creating synthetic test image...")
        return create_synthetic_test_image(output_path, size)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to expected input size (640x640) maintaining aspect ratio
    img.thumbnail(size, Image.Resampling.LANCZOS)
    
    # Create a new 640x640 image with padding if needed
    result = Image.new('RGB', size, (114, 114, 114))  # Gray padding
    
    # Center the image
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    result.paste(img, (x, y))
    
    # Save
    result.save(output_path, 'JPEG', quality=95)
    print(f"✓ Saved normalized test image to: {output_path}")
    print(f"  Size: {result.width}x{result.height}")
    
    return result

def create_synthetic_test_image(output_path, size=(640, 640)):
    """
    Create a synthetic test image with simple objects if download fails
    """
    print("Creating synthetic test image...")
    
    # Create base image (sky blue background)
    img = Image.new('RGB', size, (135, 206, 235))
    draw = ImageDraw.Draw(img)
    
    # Draw "ground" area
    ground_color = (139, 69, 19)  # Brown
    draw.rectangle([0, size[1]//2, size[0], size[1]], fill=ground_color)
    
    # Draw some "roads" (gray lines)
    road_color = (80, 80, 80)
    draw.rectangle([100, 200, 120, 600], fill=road_color)  # Vertical road
    draw.rectangle([50, 350, 500, 370], fill=road_color)   # Horizontal road
    
    # Draw some "vehicles" (colored rectangles)
    vehicles = [
        ([150, 340, 190, 365], (255, 0, 0)),    # Red car
        ([250, 340, 290, 365], (0, 0, 255)),    # Blue car
        ([350, 340, 400, 370], (0, 128, 0)),    # Green truck
        ([105, 280, 115, 310], (255, 255, 0)),  # Yellow car on vertical road
    ]
    
    for box, color in vehicles:
        draw.rectangle(box, fill=color, outline=(0, 0, 0), width=2)
    
    # Draw some "buildings" (gray rectangles)
    buildings = [
        [50, 420, 150, 520],
        [300, 450, 380, 550],
        [450, 400, 550, 530],
    ]
    
    for box in buildings:
        draw.rectangle(box, fill=(150, 150, 150), outline=(100, 100, 100), width=3)
    
    # Add text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Synthetic Aerial Test Image", fill=(255, 255, 255), font=font)
    draw.text((10, 35), f"Size: {size[0]}x{size[1]} (standard)", fill=(255, 255, 255), font=font)
    
    # Save
    img.save(output_path, 'JPEG', quality=95)
    print(f"✓ Saved synthetic test image to: {output_path}")
    print(f"  Size: {img.width}x{img.height}")
    
    return img

def main():
    # Create test_data directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    output_path = os.path.join(test_data_dir, 'aerial_test.jpg')
    
    print("Creating normalized test image for aerial detection")
    print(f"Expected size: 640x640 (standard model input)")
    print()
    
    # Try to download real aerial image first, fall back to synthetic
    try:
        create_test_image_from_url(output_path, size=(640, 640))
    except Exception as e:
        print(f"Error: {e}")
        create_synthetic_test_image(output_path, size=(640, 640))
    
    print()
    print("✓ Test image ready!")
    print(f"  Path: {output_path}")
    print(f"  Size: 640x640 pixels")
    print()
    print("You can now run:")
    print(f"  cd inference")
    print(f"  cargo run --release --features metal --example detect_video ../test_data/aerial_test.jpg")

if __name__ == '__main__':
    main()
