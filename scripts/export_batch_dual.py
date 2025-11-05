#!/usr/bin/env python3
"""
Export YOLOv8 model to ONNX format with both FP16 and FP32 variants

This script exports a YOLO model with:
- Dynamic batch size (supports batch inference for parallel tile processing)
- Two variants: FP16 (GPU/CoreML) and FP32 (CPU fallback)
- Optimized for flexible deployment scenarios

The detector will automatically select:
- FP16 model when CoreML/Metal GPU is available
- FP32 model when use_gpu=false or CoreML initialization fails
"""

import torch
from ultralytics import YOLO
import onnx
from onnx import helper, TensorProto
import sys
import os

def add_explicit_fp16_casts(model):
    """
    Add explicit Cast nodes to ensure all tensors are FP16 for CoreML compatibility.
    
    CoreML can be strict about mixed precision - this ensures any FP32 tensors
    that remain after conversion get explicit cast operations.
    """
    graph = model.graph
    
    # Find all nodes and check their outputs
    nodes_to_add = []
    outputs_to_replace = {}
    
    for node in graph.node:
        for i, output in enumerate(node.output):
            # Check if this output is used by another node
            output_type = None
            
            # Find the value_info or output that defines this tensor's type
            for value_info in list(graph.value_info) + list(graph.output):
                if value_info.name == output:
                    if value_info.type.HasField('tensor_type'):
                        output_type = value_info.type.tensor_type.elem_type
                        break
            
            # If it's FP32, add a cast to FP16
            if output_type == TensorProto.FLOAT:
                cast_output_name = f"{output}_fp16_cast"
                
                # Create Cast node: FP32 -> FP16
                cast_node = helper.make_node(
                    'Cast',
                    inputs=[output],
                    outputs=[cast_output_name],
                    to=TensorProto.FLOAT16,
                    name=f"{node.name}_fp16_cast"
                )
                
                nodes_to_add.append(cast_node)
                outputs_to_replace[output] = cast_output_name
    
    # Update all nodes that consume the casted outputs
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in outputs_to_replace:
                node.input[i] = outputs_to_replace[input_name]
    
    # Add the cast nodes to the graph
    graph.node.extend(nodes_to_add)
    
    # Update value_infos to reflect FP16 types
    for value_info in graph.value_info:
        if value_info.name in outputs_to_replace:
            # Create new FP16 value_info
            new_value_info = helper.make_tensor_value_info(
                outputs_to_replace[value_info.name],
                TensorProto.FLOAT16,
                [dim.dim_value if dim.HasField('dim_value') else None 
                 for dim in value_info.type.tensor_type.shape.dim]
            )
            graph.value_info.append(new_value_info)
    
    print(f"    ✓ Added {len(nodes_to_add)} explicit Cast nodes for FP16 compatibility")
    
    return model

def export_model_variant(model, output_path: str, fp16: bool = False):
    """Export a single model variant (FP16 or FP32)"""
    variant_name = "FP16" if fp16 else "FP32"
    print(f"\n📤 Exporting {variant_name} variant...")
    
    # Export with dynamic batch size (always as FP32 first)
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=12,
        dynamic=True,  # Enable dynamic batch size
        half=False,    # Export as FP32 (will convert to FP16 via ONNX if needed)
    )
    
    # The exported file will be named based on the model
    base_name = os.path.splitext(os.path.basename(model.ckpt_path))[0]
    onnx_file = f"{base_name}.onnx"
    
    # Verify the exported model has dynamic batch dimension
    print(f"  🔍 Verifying dynamic batch dimension...")
    onnx_model = onnx.load(onnx_file)
    
    # Check input shape
    input_shape = onnx_model.graph.input[0].type.tensor_type.shape
    batch_dim = input_shape.dim[0]
    
    if batch_dim.HasField('dim_param'):
        print(f"  ✅ Dynamic batch dimension confirmed: {batch_dim.dim_param}")
    elif batch_dim.HasField('dim_value'):
        print(f"  ⚠️  Fixed batch dimension: {batch_dim.dim_value}")
    
    print(f"  Input shape: ", end="")
    for i, dim in enumerate(input_shape.dim):
        if dim.HasField('dim_param'):
            print(f"{dim.dim_param}", end="")
        elif dim.HasField('dim_value'):
            print(f"{dim.dim_value}", end="")
        if i < len(input_shape.dim) - 1:
            print(" × ", end="")
    print()
    
    # Convert to FP16 if requested
    if fp16:
        print(f"  🔄 Converting to FP16...")
        from onnxconverter_common import float16
        
        # Convert all float32 tensors to float16 with explicit casts
        # This ensures ALL tensors (including intermediate ones) are FP16
        onnx_model_fp16 = float16.convert_float_to_float16(
            onnx_model,
            keep_io_types=False,  # Convert inputs/outputs to FP16 too
            disable_shape_infer=False,  # Keep shape inference enabled
        )
        
        # Post-process: Add explicit Cast nodes for any remaining FP32 tensors
        print(f"  🔧 Adding explicit FP16 casts for CoreML compatibility...")
        onnx_model_fp16 = add_explicit_fp16_casts(onnx_model_fp16)
        
        # Save the FP16 model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(onnx_model_fp16, output_path)
        
        # Clean up the intermediate FP32 file
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
    else:
        # Move to output path (FP32)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(onnx_file):
            os.rename(onnx_file, output_path)
    
    if os.path.exists(output_path):
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✅ {variant_name} model exported successfully!")
        print(f"  📁 Output: {output_path}")
        print(f"  📊 Model size: {file_size:.1f} MB")
        return True
    else:
        print(f"  ❌ Export failed - file not found: {output_path}")
        return False

def export_dual_models(model_path: str, output_base_name: str = "yolov8m_batch"):
    print("📦 Exporting YOLO Model - Dual Variants (FP16 + FP32)")
    print("=" * 70)
    print(f"Input model: {model_path}")
    print(f"Output base: ../models/{output_base_name}_{{fp16|fp32}}.onnx")
    
    # Load the model
    if model_path.endswith('.pt'):
        print("\n🔄 Loading YOLOv8 .pt model...")
        model = YOLO(model_path)
        
    elif model_path.endswith('.pth'):
        print("\n⚠️  .pth checkpoint detected")
        print("Attempting to load as YOLOv8 checkpoint...")
        
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
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease provide a YOLOv8 .pt model file instead of .pth checkpoint")
            return False
    
    else:
        print(f"❌ Unsupported file format: {model_path}")
        return False
    
    # Export FP16 variant (for GPU/CoreML)
    fp16_path = f"../models/{output_base_name}_fp16.onnx"
    fp16_success = export_model_variant(model, fp16_path, fp16=True)
    
    # Re-load model for FP32 export (ultralytics caches export config)
    if model_path.endswith('.pt'):
        model = YOLO(model_path)
    
    # Export FP32 variant (for CPU fallback)
    fp32_path = f"../models/{output_base_name}_fp32.onnx"
    fp32_success = export_model_variant(model, fp32_path, fp16=False)
    
    # Summary
    print("\n" + "=" * 70)
    if fp16_success and fp32_success:
        print("✅ BOTH VARIANTS EXPORTED SUCCESSFULLY!")
        print("\n📋 Usage in detector config:")
        print(f"  fp16_model_path: \"{output_base_name}_fp16.onnx\"  // GPU (CoreML/Metal)")
        print(f"  fp32_model_path: \"{output_base_name}_fp32.onnx\"  // CPU fallback")
        print("\n💡 The detector will automatically select:")
        print("  • FP16 when use_gpu=true and CoreML is available")
        print("  • FP32 when use_gpu=false or CoreML initialization fails")
        print("\n📋 Next steps:")
        print("  1. Update DetectorConfig with both model paths")
        print("  2. Test GPU path: cargo run --release --features metal")
        print("  3. Test CPU fallback: cargo run --release (no metal feature)")
        return True
    else:
        print("❌ EXPORT FAILED - check errors above")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_batch_dual.py <model_path.pt> [output_base_name]")
        print("\nExample:")
        print("  python export_batch_dual.py ~/Downloads/m_stage1-7e1e5299.pth yolov8m_batch")
        print("\nOutputs:")
        print("  ../models/yolov8m_batch_fp16.onnx  (for GPU)")
        print("  ../models/yolov8m_batch_fp32.onnx  (for CPU)")
        sys.exit(1)
    
    model_path = os.path.expanduser(sys.argv[1])
    output_base_name = sys.argv[2] if len(sys.argv) > 2 else "yolov8m_batch"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    success = export_dual_models(model_path, output_base_name)
    sys.exit(0 if success else 1)
