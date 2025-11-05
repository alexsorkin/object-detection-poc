#!/usr/bin/env python3
"""
Export YOLOv8 model to ONNX format with dynamic batch size and FP16 optimization

This script exports a YOLO model with:
- Dynamic batch size (supports batch inference for parallel tile processing)
- FP16 quantization for reduced model size and faster inference
- Optimized for CoreML/Metal execution on macOS
"""

import torch
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import sys
import os

def export_model_with_dynamic_batch(model_path: str, output_name: str = "yolov8m_batch_fp16"):
    print("📦 Exporting YOLO Model with Dynamic Batch Size and FP16")
    print("=" * 60)
    print(f"Input model: {model_path}")
    
    # Load the model
    if model_path.endswith('.pt'):
        print("\n🔄 Loading YOLOv8 .pt model...")
        model = YOLO(model_path)
        
        print("\n📤 Exporting to ONNX with dynamic batch size...")
        # Export with dynamic batch size
        model.export(
            format="onnx",
            imgsz=640,
            simplify=True,
            opset=12,
            dynamic=True,  # Enable dynamic batch size
            half=True,     # Export as FP16
        )
        
        # The exported file will be named based on the input .pt file
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        onnx_file = f"{base_name}.onnx"
        
    elif model_path.endswith('.pth'):
        print("\n⚠️  .pth checkpoint detected - this requires model architecture definition")
        print("This appears to be a raw PyTorch checkpoint, not a YOLOv8 .pt file")
        print("\nOptions:")
        print("1. If this is a YOLOv8 model, load it with YOLO() first")
        print("2. If this is a custom model, you need to define the architecture")
        print("\nAttempting to load as YOLOv8 checkpoint...")
        
        # Try to load as YOLO checkpoint
        try:
            # Create a YOLO model and load the checkpoint
            model = YOLO('yolov8m.pt')  # Start with base architecture
            
            # Load the checkpoint weights
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Try to extract the model state
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict into model
            model.model.load_state_dict(state_dict, strict=False)
            
            print("\n📤 Exporting to ONNX with dynamic batch size...")
            model.export(
                format="onnx",
                imgsz=640,
                simplify=True,
                opset=12,
                dynamic=True,
                half=True,
            )
            
            onnx_file = "yolov8m.onnx"
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease provide a YOLOv8 .pt model file instead of .pth checkpoint")
            return False
    
    else:
        print(f"❌ Unsupported file format: {model_path}")
        return False
    
    # Verify the exported model has dynamic batch dimension
    print("\n🔍 Verifying dynamic batch dimension...")
    onnx_model = onnx.load(onnx_file)
    
    # Check input shape
    input_shape = onnx_model.graph.input[0].type.tensor_type.shape
    batch_dim = input_shape.dim[0]
    
    if batch_dim.HasField('dim_param'):
        print(f"✅ Dynamic batch dimension confirmed: {batch_dim.dim_param}")
    elif batch_dim.HasField('dim_value'):
        print(f"⚠️  Fixed batch dimension: {batch_dim.dim_value}")
    else:
        print("⚠️  Could not determine batch dimension type")
    
    print(f"\nInput shape: ", end="")
    for i, dim in enumerate(input_shape.dim):
        if dim.HasField('dim_param'):
            print(f"{dim.dim_param}", end="")
        elif dim.HasField('dim_value'):
            print(f"{dim.dim_value}", end="")
        if i < len(input_shape.dim) - 1:
            print(" × ", end="")
    print()
    
    # Rename to output name
    output_file = f"../models/{output_name}.onnx"
    os.makedirs("../models", exist_ok=True)
    
    if os.path.exists(onnx_file):
        os.rename(onnx_file, output_file)
        print(f"\n✅ Model exported successfully!")
        print(f"📁 Output: {output_file}")
        
        # Get file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 Model size: {file_size:.1f} MB")
        
        print("\n📋 Next steps:")
        print(f"  1. Update detector config to use: {output_name}.onnx")
        print("  2. Test with: cargo run --release --features metal --example detect_tile_overlap")
        print("  3. Verify batch processing works with multiple tiles")
        
        return True
    else:
        print(f"\n❌ Export failed - file not found: {onnx_file}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_batch_fp16.py <model_path.pt> [output_name]")
        print("\nExample:")
        print("  python export_batch_fp16.py ~/Downloads/m_stage1-7e1e5299.pth yolov8m_batch_fp16")
        sys.exit(1)
    
    model_path = os.path.expanduser(sys.argv[1])
    output_name = sys.argv[2] if len(sys.argv) > 2 else "yolov8m_batch_fp16"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    success = export_model_with_dynamic_batch(model_path, output_name)
    sys.exit(0 if success else 1)
