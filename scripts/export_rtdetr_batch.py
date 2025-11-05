#!/usr/bin/env python3
"""
Export RT-DETR model with dynamic batch size support for parallel tile processing.

This script exports RT-DETR models from Hugging Face with dynamic batch dimensions,
allowing efficient batch inference on multiple tiles simultaneously.

Usage:
    python export_rtdetr_batch.py
"""

import torch
import onnx
import onnxruntime as ort
from transformers import RTDetrForObjectDetection
import os
import warnings

# Suppress TracerWarnings during export
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

def export_rtdetr_with_dynamic_batch(model_name="PekingU/rtdetr_r50vd_coco_o365", output_path="../models/rtdetr_batch.onnx"):
    """
    Export RT-DETR model with dynamic batch size.
    
    Args:
        model_name: Hugging Face model name
        output_path: Path to save the ONNX model
    """
    
    print(f"Loading RT-DETR model: {model_name}")
    print("Note: TracerWarnings during export are expected and can be safely ignored.")
    print("These occur due to dynamic control flow in the transformer architecture.\n")
    
    model = RTDetrForObjectDetection.from_pretrained(model_name)
    model.eval()
    
    # Move to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Create dummy input with batch size 1
    batch_size = 1
    input_size = 640  # RT-DETR typically uses 640x640
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    print(f"\nExporting model with dynamic batch size...")
    print(f"Input shape: [batch, 3, {input_size}, {input_size}]")
    
    # Export with dynamic axes for batch dimension
    # Use torch.onnx.export with strict=False to handle dynamic control flow
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
        warnings.filterwarnings('ignore', message='.*Converting a tensor to a Python.*')
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,  # Use latest stable opset
            do_constant_folding=True,
            input_names=['images'],
            output_names=['pred_logits', 'pred_boxes'],
            dynamic_axes={
                'images': {0: 'batch'},      # Dynamic batch size for input
                'pred_logits': {0: 'batch'},  # Dynamic batch size for logits output
                'pred_boxes': {0: 'batch'}    # Dynamic batch size for boxes output
            },
            verbose=False
        )
    
    print(f"✓ Model exported to: {output_path}")
    
    # Verify the exported model
    print("\nVerifying exported model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Print input/output shapes
    print("\nModel signature:")
    for input_tensor in onnx_model.graph.input:
        print(f"  Input: {input_tensor.name}")
        shape = [dim.dim_param if dim.dim_param else dim.dim_value 
                for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"    Shape: {shape}")
    
    for output_tensor in onnx_model.graph.output:
        print(f"  Output: {output_tensor.name}")
        shape = [dim.dim_param if dim.dim_param else dim.dim_value 
                for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"    Shape: {shape}")
    
    # Test with different batch sizes
    print("\nTesting dynamic batch inference...")
    test_batch_sizes = [1, 2, 4]
    
    sess = ort.InferenceSession(output_path)
    
    for batch in test_batch_sizes:
        test_input = torch.randn(batch, 3, input_size, input_size).numpy()
        outputs = sess.run(None, {'images': test_input})
        
        # RT-DETR outputs: pred_logits, pred_boxes (first 2 are the main outputs)
        pred_logits = outputs[0]
        pred_boxes = outputs[1]
        print(f"  Batch {batch}: logits {pred_logits.shape}, boxes {pred_boxes.shape}")
    
    print("\n✅ Model successfully exported with dynamic batch support!")
    print(f"\nModel file size: {os.path.getsize(output_path) / (1024**2):.1f} MB")


def export_rtdetr_v2_r18vd():
    """Export RT-DETR V2-R18vd (ResNet-18 backbone, smallest/fastest)"""
    export_rtdetr_with_dynamic_batch(
        model_name="PekingU/rtdetr_v2_r18vd",
        output_path="../models/rtdetr_v2_r18vd_batch.onnx"
    )


def export_rtdetr_v2_r34vd():
    """Export RT-DETR V2-R34vd (ResNet-34 backbone, balanced size/speed)"""
    export_rtdetr_with_dynamic_batch(
        model_name="PekingU/rtdetr_v2_r34vd",
        output_path="../models/rtdetr_v2_r34vd_batch.onnx"
    )


def export_rtdetr_v2_r50vd():
    """Export RT-DETR V2-R50vd (ResNet-50 backbone, V2 version)"""
    export_rtdetr_with_dynamic_batch(
        model_name="PekingU/rtdetr_v2_r50vd",
        output_path="../models/rtdetr_v2_r50vd_batch.onnx"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export RT-DETR with dynamic batch size")
    parser.add_argument(
        "--model",
        choices=["v2_r50vd", "v2_r34vd", "v2_r18vd", "all"],
        default="v2_r50vd",
        help="Which RT-DETR variant to export (default: v2_r50vd)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RT-DETR Dynamic Batch Export")
    print("=" * 70)
    
    if args.model == "v2_r18vd":
        print("\n📦 Exporting RT-DETR V2-R18vd (Smallest/Fastest)...")
        export_rtdetr_v2_r18vd()
    
    if args.model == "v2_r34vd":
        print("\n📦 Exporting RT-DETR V2-R34vd (Balanced)...")
        export_rtdetr_v2_r34vd()

    if args.model == "v2_r50vd":
        print("\n📦 Exporting RT-DETR V2-R50vd (Large/Slow)...")
        export_rtdetr_v2_r50vd()
    
    print("\n✅ All exports complete!")
    print("\nNext steps:")
    print("  1. Update DetectorConfig to use the new model")
    print("  2. Test with: cargo run --release --features metal --example detect_pipeline -- --detector rtdetr")
    print("  3. Verify batch execution logs show GPU batch (not sequential)")
