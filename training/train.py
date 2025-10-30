#!/usr/bin/env python3
"""
Military Target Detection Training Script

This script trains a YOLOv8 model to detect military targets in video streams.
Optimized for real-time inference and cross-platform deployment.
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilitaryTargetTrainer:
    """Military target detection model trainer using YOLOv8"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize trainer with configuration"""
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.results = None
        
        # Detect and configure best available device
        self.device = self.detect_best_device()
        
        # Create output directories
        self.setup_directories()
        
    def load_config(self):
        """Load training configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def detect_best_device(self):
        """Detect and configure the best available training device"""
        device_config = self.config.get('device', 'auto')
        
        # If device is explicitly set (not auto), use it
        if device_config != 'auto':
            logger.info(f"Using explicitly configured device: {device_config}")
            return device_config
        
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = "0"  # Use first CUDA GPU
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {gpu_name}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Use Apple Metal Performance Shaders
            logger.info("Apple Metal Performance Shaders (MPS) detected")
            logger.info("Mac GPU acceleration enabled")
            
            # Log Mac-specific optimizations
            if hasattr(torch.backends.mps, 'is_built'):
                logger.info(f"MPS built: {torch.backends.mps.is_built()}")
        else:
            device = "cpu"
            logger.info("Using CPU for training (GPU not available)")
            logger.warning("Training will be significantly slower on CPU")
        
        return device
    
    def apply_mac_optimizations(self):
        """Apply Mac-specific training optimizations for MPS"""
        logger.info("Applying Mac MPS optimizations...")
        
        try:
            # Clear MPS cache before training
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("Cleared MPS cache")
                
            # Set memory management
            mac_config = self.config.get('mac_optimizations', {})
            
            # Enable memory-efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.info("Memory-efficient attention available")
                
            # Log memory information
            import psutil
            memory = psutil.virtual_memory()
            logger.info(f"System memory: {memory.total // (1024**3)}GB total, "
                       f"{memory.available // (1024**3)}GB available")
                       
        except Exception as e:
            logger.warning(f"Could not apply all Mac optimizations: {e}")
    
    def setup_directories(self):
        """Create necessary directories for training"""
        directories = [
            "../models",
            "../data/images/train", 
            "../data/images/val",
            "../data/images/test",
            "../data/labels/train",
            "../data/labels/val", 
            "../data/labels/test",
            "runs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created training directories")
    
    def validate_dataset(self):
        """Validate dataset structure and files"""
        data_path = Path(self.config['data']['path'])
        
        if not data_path.exists():
            logger.error(f"Dataset path {data_path} does not exist")
            return False
            
        # Check for required directories
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for req_dir in required_dirs:
            dir_path = data_path / req_dir
            if not dir_path.exists():
                logger.warning(f"Directory {dir_path} does not exist")
                return False
                
        # Count images and labels
        train_images = len(list((data_path / 'images/train').glob('*.jpg'))) + \
                      len(list((data_path / 'images/train').glob('*.png')))
        train_labels = len(list((data_path / 'labels/train').glob('*.txt')))
        
        val_images = len(list((data_path / 'images/val').glob('*.jpg'))) + \
                    len(list((data_path / 'images/val').glob('*.png')))
        val_labels = len(list((data_path / 'labels/val').glob('*.txt')))
        
        logger.info(f"Training: {train_images} images, {train_labels} labels")
        logger.info(f"Validation: {val_images} images, {val_labels} labels")
        
        if train_images == 0 or val_images == 0:
            logger.error("No training or validation images found")
            return False
            
        return True
    
    def initialize_model(self):
        """Initialize YOLOv8 model"""
        model_name = self.config.get('model', 'yolov8n.pt')
        
        try:
            self.model = YOLO(model_name)
            logger.info(f"Initialized YOLOv8 model: {model_name}")
            
            # Print model info
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            logger.info(f"Model size: {os.path.getsize(model_name) / 1024 / 1024:.1f}MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def train(self):
        """Train the military target detection model"""
        if not self.validate_dataset():
            logger.error("Dataset validation failed")
            return None
            
        self.initialize_model()
        
        # Training parameters from config
        train_params = {
            'data': self.config_path,
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch_size', 16),
            'imgsz': self.config.get('img_size', 640),
            'workers': self.config.get('workers', 8),
            'patience': self.config.get('patience', 20),
            'device': self.device,
            'project': 'runs/detect',
            'name': f'military_targets_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'save_period': self.config.get('save_period', 10),
            'plots': self.config.get('plots', True),
            'val': self.config.get('val', True)
        }
        
        # Add optimization parameters
        if 'optimizer' in self.config:
            train_params['optimizer'] = self.config['optimizer']
        if 'lr0' in self.config:
            train_params['lr0'] = self.config['lr0']
        if 'momentum' in self.config:
            train_params['momentum'] = self.config['momentum']
        if 'weight_decay' in self.config:
            train_params['weight_decay'] = self.config['weight_decay']
            
        logger.info("Starting training with parameters:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")
        
        try:
            # Apply Mac-specific optimizations if using MPS
            if self.device == "mps":
                self.apply_mac_optimizations()
            
            # Start training
            self.results = self.model.train(**train_params)
            logger.info("Training completed successfully")
            
            # Save best model for inference
            best_model_path = "../models/military_targets_best.pt"
            self.model.save(best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def export_model(self, export_format='onnx'):
        """Export trained model for deployment"""
        if self.model is None:
            logger.error("No trained model available for export")
            return None
            
        export_params = self.config.get('export', {})
        export_format = export_params.get('format', export_format)
        
        try:
            # Load best model for export
            best_model = YOLO("../models/military_targets_best.pt")
            
            export_path = best_model.export(
                format=export_format,
                dynamic=export_params.get('dynamic', True),
                simplify=export_params.get('simplify', True),
                opset=export_params.get('opset', 11)
            )
            
            logger.info(f"Model exported to: {export_path}")
            
            # Copy to standard location
            if export_format == 'onnx':
                import shutil
                final_path = "../models/military_targets.onnx"
                shutil.copy2(export_path, final_path)
                logger.info(f"Model copied to: {final_path}")
                
            return export_path
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise
    
    def validate_model(self):
        """Validate trained model on test set"""
        if self.model is None:
            logger.error("No trained model available for validation")
            return None
            
        try:
            # Validate on test set
            test_results = self.model.val(
                data=self.config_path,
                split='test',
                save_json=True,
                plots=True
            )
            
            logger.info("Model validation completed")
            logger.info(f"mAP50: {test_results.box.map50:.3f}")
            logger.info(f"mAP50-95: {test_results.box.map:.3f}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return None

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train military target detection model')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--export', default='onnx', help='Export format (onnx, torchscript, etc.)')
    parser.add_argument('--validate', action='store_true', help='Validate model after training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MilitaryTargetTrainer(args.config)
    
    # Train model
    logger.info("Starting military target detection training")
    results = trainer.train()
    
    if results is None:
        logger.error("Training failed")
        sys.exit(1)
    
    # Export model
    logger.info(f"Exporting model in {args.export} format")
    export_path = trainer.export_model(args.export)
    
    # Optional validation
    if args.validate:
        logger.info("Validating trained model")
        val_results = trainer.validate_model()
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()