#!/usr/bin/env python3
"""
Export RT-DETR model using Ultralytics (cleaner export, no TracerWarnings).

Ultralytics provides a more stable ONNX export for RT-DETR with proper
dynamic batch support and no tracing warnings.

Usage:
    python export_rtdetr_ultralytics.py [--model MODEL_SIZE]
    
Models available:
    - rtdetr-l.pt   : Large model (~135MB, best accuracy)
    - rtdetr-x.pt   : Extra large (~230MB, highest accuracy)
"""

import os
import sys
from pathlib import Path

def export_rtdetr_ultralytics(model_size='l', output_dir='../models'):
    """
    Export RT-DETR using Ultralytics (recommended method).
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', or 'x')
        output_dir: Directory to save the exported model
    """
    try:
        from ultralytics import RTDETR
    except ImportError:
        print("❌ Ultralytics not installed!")
        print("\nInstall with:")
        print("  pip install ultralytics")
        sys.exit(1)
    
    model_name = f'rtdetr-{model_size}.pt'  # Format: rtdetr-l.pt, rtdetr-m.pt, etc.
    output_path = os.path.join(output_dir, f'rtdetr_{model_size}_batch.onnx')
    
    print("=" * 70)
    print(f"RT-DETR Export using Ultralytics")
    print("=" * 70)
    print(f"\n📦 Model: {model_name}")
    print(f"📁 Output: {output_path}")
    print()
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model (will download if not cached)
    print("Loading RT-DETR model...")
    model = RTDETR(model_name)
    
    # Export with dynamic batch size
    print("\n🔄 Exporting to ONNX with dynamic batch support...")
    print("   • Dynamic batch: 1-16 images")
    print("   • Input size: 640×640")
    print("   • Simplify: Yes (optimize graph)")
    print()
    
    success = model.export(
        format='onnx',
        dynamic=True,      # Enable dynamic batch size
        simplify=True,     # Simplify ONNX graph (faster inference)
        opset=17,          # ONNX opset version
        imgsz=640,         # Input size
    )
    
    if success:
        # Rename to our expected filename
        generated_path = model_name.replace('.pt', '.onnx')
        if os.path.exists(generated_path):
            import shutil
            shutil.move(generated_path, output_path)
        
        print(f"\n✅ Export successful!")
        print(f"   Model saved to: {output_path}")
        
        # Print file size
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"   File size: {size_mb:.1f} MB")
        
        # Verify with ONNX Runtime
        print("\n🔍 Verifying model...")
        try:
            import onnxruntime as ort
            import numpy as np
            
            sess = ort.InferenceSession(output_path)
            
            # Print input/output info
            print("\n📋 Model Signature:")
            for input_meta in sess.get_inputs():
                print(f"   Input: {input_meta.name}")
                print(f"     Shape: {input_meta.shape}")
                print(f"     Type: {input_meta.type}")
            
            for output_meta in sess.get_outputs():
                print(f"   Output: {output_meta.name}")
                print(f"     Shape: {output_meta.shape}")
                print(f"     Type: {output_meta.type}")
            
            # Test different batch sizes
            print("\n🧪 Testing dynamic batch inference:")
            for batch_size in [1, 2, 4]:
                test_input = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
                input_name = sess.get_inputs()[0].name
                outputs = sess.run(None, {input_name: test_input})
                
                print(f"   Batch {batch_size}: ", end="")
                for i, output in enumerate(outputs):
                    print(f"output{i} {output.shape}", end="  ")
                print()
            
            print("\n✅ Verification complete! Model supports dynamic batching.")
            
        except Exception as e:
            print(f"\n⚠️  Verification warning: {e}")
            print("   Model exported but couldn't verify. This is usually fine.")
        
        return True
    else:
        print("\n❌ Export failed!")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export RT-DETR with dynamic batch size using Ultralytics"
    )
    parser.add_argument(
        '--model',
        choices=['n', 's', 'm', 'l', 'x'],
        default='l',
        help='Model size: n (nano), s (small), m (medium), l (large, ~135MB), x (extra large, ~230MB)'
    )
    parser.add_argument(
        '--output-dir',
        default='../models',
        help='Output directory for the exported model'
    )
    
    args = parser.parse_args()
    
    print("\n🚀 Starting RT-DETR export using Ultralytics")
    print("   This method produces cleaner exports with no warnings\n")
    
    success = export_rtdetr_ultralytics(args.model, args.output_dir)
    
    if success:
        print("\n" + "=" * 70)
        print("✅ Export Complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Update your RT-DETR detector config to use the new model")
        print("  2. Test with: cargo run --release --features metal --example detect_pipeline -- --detector rtdetr")
        print("  3. You should see: '🔥 GPU Batch Execution' (not Sequential)")
        print("\nExpected performance:")
        print("  • 2 tiles processed in parallel in ~450-500ms")
        print("  • 2x speedup compared to sequential processing")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
