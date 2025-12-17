#!/usr/bin/env python3
"""
Normalize test image to expected input size (640x640)
Current size: 1120x787 -> Target: 640x640
"""

from PIL import Image
import os

def normalize_image(input_path, output_path, target_size=(640, 640)):
    """
    Resize image to model input size with aspect ratio preservation and padding
    """
    print(f"Normalizing image: {input_path}")
    
    # Load image
    img = Image.open(input_path)
    original_size = img.size
    print(f"  Original size: {original_size[0]}x{original_size[1]}")
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Calculate scaling to fit within target size while maintaining aspect ratio
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    scaled_size = img.size
    print(f"  Scaled size: {scaled_size[0]}x{scaled_size[1]}")
    
    # Create new image with target size and gray padding (114, 114, 114)
    # Standard padding used during inference
    result = Image.new('RGB', target_size, (114, 114, 114))
    
    # Center the scaled image
    x = (target_size[0] - scaled_size[0]) // 2
    y = (target_size[1] - scaled_size[1]) // 2
    result.paste(img, (x, y))
    
    # Save normalized image
    result.save(output_path, 'JPEG', quality=95)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Final size: {result.width}x{result.height}")
    
    return result

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_dir = os.path.join(script_dir, '..', 'inference')
    test_data_dir = os.path.join(inference_dir, 'test_data')
    
    input_path = os.path.join(test_data_dir, 'aerial_test.jpg')
    output_path = os.path.join(test_data_dir, 'aerial_test_640.jpg')
    
    if not os.path.exists(input_path):
        print(f"Error: Input image not found: {input_path}")
        return 1
    
    print("Image Normalization")
    print("=" * 50)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target: 640x640 (standard input size)")
    print()
    
    # Normalize
    normalize_image(input_path, output_path, target_size=(640, 640))
    
    print()
    print("✓ Normalization complete!")
    print()
    print("Note: The original preprocessing in the detector will resize images")
    print("automatically, but using pre-normalized images can slightly improve")
    print("inference speed and reduce memory usage.")
    print()
    print("Test with normalized image:")
    print(f"  cd {inference_dir}")
    print(f"  cargo run --release --features metal --example detect_video test_data/aerial_test_640.jpg")
    
    return 0

if __name__ == '__main__':
    exit(main())
