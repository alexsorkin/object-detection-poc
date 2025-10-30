#!/usr/bin/env python3
"""
Model export utility for military target detection
Converts trained PyTorch models to ONNX format for deployment
"""

import os
import sys
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class ModelExporter:
    """Export trained models for deployment"""
    
    def __init__(self, model_path: str):
        """Initialize exporter with trained model"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        
    def export_to_onnx(self, 
                       output_path: str = None, 
                       input_size: tuple = (640, 640),
                       dynamic_axes: bool = True,
                       opset_version: int = 11,
                       simplify: bool = True):
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            input_size: Input image size (height, width)
            dynamic_axes: Enable dynamic input shapes
            opset_version: ONNX opset version
            simplify: Simplify ONNX graph
        """
        if output_path is None:
            output_path = self.model_path.replace('.pt', '.onnx')
            
        logger.info(f"Exporting model to ONNX format")
        logger.info(f"  Input model: {self.model_path}")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Dynamic axes: {dynamic_axes}")
        logger.info(f"  Opset version: {opset_version}")
        
        try:
            # Export using ultralytics
            onnx_path = self.model.export(
                format='onnx',
                dynamic=dynamic_axes,
                simplify=simplify,
                opset=opset_version,
                imgsz=input_size
            )
            
            # Move to desired location if different
            if onnx_path != output_path:
                import shutil
                shutil.move(onnx_path, output_path)
                logger.info(f"Model moved to: {output_path}")
            
            # Verify ONNX model
            self._verify_onnx_model(output_path, input_size)
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _verify_onnx_model(self, onnx_path: str, input_size: tuple):
        """Verify ONNX model can be loaded and run"""
        logger.info("Verifying ONNX model...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model structure is valid")
            
            # Test inference
            session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()
            
            logger.info(f"  Input name: {input_info.name}")
            logger.info(f"  Input shape: {input_info.shape}")
            logger.info(f"  Input type: {input_info.type}")
            
            for i, output in enumerate(output_info):
                logger.info(f"  Output {i}: {output.name}, shape: {output.shape}")
            
            # Test with dummy data
            dummy_input = np.random.rand(1, 3, *input_size).astype(np.float32)
            outputs = session.run(None, {input_info.name: dummy_input})
            
            logger.info(f"✓ ONNX inference test passed")
            logger.info(f"  Output shapes: {[out.shape for out in outputs]}")
            
            # Model size
            model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info(f"  Model size: {model_size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise
    
    def export_to_tensorrt(self, output_path: str = None, input_size: tuple = (640, 640)):
        """Export model to TensorRT format (requires TensorRT)"""
        if output_path is None:
            output_path = self.model_path.replace('.pt', '.engine')
            
        logger.info(f"Exporting model to TensorRT format")
        
        try:
            # Export using ultralytics
            engine_path = self.model.export(
                format='engine',
                imgsz=input_size,
                workspace=4,  # 4GB workspace
                verbose=True
            )
            
            if engine_path != output_path:
                import shutil
                shutil.move(engine_path, output_path)
                
            logger.info(f"TensorRT model exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            logger.error("Make sure TensorRT is installed and GPU is available")
            raise
    
    def export_to_torchscript(self, output_path: str = None, input_size: tuple = (640, 640)):
        """Export model to TorchScript format"""
        if output_path is None:
            output_path = self.model_path.replace('.pt', '_torchscript.pt')
            
        logger.info(f"Exporting model to TorchScript format")
        
        try:
            # Export using ultralytics
            torchscript_path = self.model.export(
                format='torchscript',
                imgsz=input_size
            )
            
            if torchscript_path != output_path:
                import shutil
                shutil.move(torchscript_path, output_path)
                
            logger.info(f"TorchScript model exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
    
    def create_model_info(self, output_dir: str):
        """Create model information file for deployment"""
        model_info = {
            'model_name': 'military_target_detector',
            'version': '1.0.0',
            'framework': 'YOLOv8',
            'input_size': [640, 640],
            'input_format': 'RGB',
            'input_range': [0.0, 1.0],
            'classes': [
                {'id': 0, 'name': 'armed_personnel', 'color': [255, 0, 0]},
                {'id': 1, 'name': 'rocket_launcher', 'color': [0, 255, 0]},
                {'id': 2, 'name': 'military_vehicle', 'color': [0, 0, 255]},
                {'id': 3, 'name': 'heavy_weapon', 'color': [255, 255, 0]}
            ],
            'preprocessing': {
                'resize': True,
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'postprocessing': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.45,
                'max_detections': 100
            }
        }
        
        import json
        info_path = os.path.join(output_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
            
        logger.info(f"Model info saved: {info_path}")
        return info_path

def main():
    """Main export function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export military target detection model')
    parser.add_argument('model', help='Path to trained PyTorch model (.pt)')
    parser.add_argument('--output-dir', default='../models', help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['onnx'], 
                       choices=['onnx', 'tensorrt', 'torchscript'], 
                       help='Export formats')
    parser.add_argument('--input-size', nargs=2, type=int, default=[640, 640], 
                       help='Input image size (height width)')
    parser.add_argument('--dynamic', action='store_true', 
                       help='Enable dynamic input shapes for ONNX')
    parser.add_argument('--opset', type=int, default=11, 
                       help='ONNX opset version')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize exporter
    exporter = ModelExporter(args.model)
    
    # Export in requested formats
    for fmt in args.formats:
        output_path = os.path.join(args.output_dir, f'military_targets.{fmt}')
        
        if fmt == 'onnx':
            logger.info(f"Exporting to ONNX format...")
            exporter.export_to_onnx(
                output_path.replace('.onnx', '.onnx'),
                tuple(args.input_size),
                args.dynamic,
                args.opset
            )
        elif fmt == 'tensorrt':
            logger.info(f"Exporting to TensorRT format...")
            exporter.export_to_tensorrt(
                output_path.replace('.tensorrt', '.engine'),
                tuple(args.input_size)
            )
        elif fmt == 'torchscript':
            logger.info(f"Exporting to TorchScript format...")
            exporter.export_to_torchscript(
                output_path.replace('.torchscript', '.pt'),
                tuple(args.input_size)
            )
    
    # Create model info file
    exporter.create_model_info(args.output_dir)
    
    logger.info("Model export completed successfully")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()