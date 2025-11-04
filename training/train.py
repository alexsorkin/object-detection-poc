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
        # Handle both nested config and direct data path
        if isinstance(self.config['data'], dict):
            # Config file format: data: {path: ..., train: ...}
            data_path = Path(self.config['data']['path'])
            # For nested config, check if path exists
            if not data_path.exists():
                logger.error(f"Dataset path {data_path} does not exist")
                return False
        else:
            # Direct YAML path format: data: path/to/dataset.yaml
            data_yaml = Path(self.config['data'])
            if not data_yaml.exists():
                logger.error(f"Dataset YAML {data_yaml} does not exist")
                return False
            
            # Load and validate the dataset YAML
            with open(data_yaml, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['path', 'train', 'nc', 'names']
            for field in required_fields:
                if field not in dataset_config:
                    logger.error(f"Dataset YAML missing required field: {field}")
                    return False
            
            data_path = Path(dataset_config['path'])
            if not data_path.exists():
                logger.error(f"Dataset path {data_path} does not exist")
                return False
            
            # Check if train path exists (relative to data_path)
            train_path = data_path / dataset_config['train']
            if not train_path.exists():
                logger.error(f"Training data path {train_path} does not exist")
                return False
            
            # Count training images
            train_images = len(list(train_path.glob('*.jpg'))) + \
                          len(list(train_path.glob('*.png')))
            
            if train_images == 0:
                logger.error(f"No training images found in {train_path}")
                return False
            
            logger.info(f"Found {train_images} training images in {train_path}")
        
        logger.info("Dataset validation passed")
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
        # Use the data path directly (it should be the dataset YAML path)
        data_path = self.config.get('data', self.config_path)
        
        train_params = {
            'data': data_path,
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch', 16),
            'imgsz': self.config.get('imgsz', 640),
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
            
            # Save best model for inference with both names for compatibility
            models_dir = Path("../models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            best_model_path = models_dir / "military_target_detector.pt"
            alt_model_path = models_dir / "military_targets_best.pt"
            
            # Copy the trained model
            import shutil
            best_weights = Path(train_params['project']) / train_params['name'] / 'weights' / 'best.pt'
            shutil.copy(best_weights, best_model_path)
            shutil.copy(best_weights, alt_model_path)
            
            logger.info(f"Saved best model to {best_model_path}")
            logger.info(f"Also saved to {alt_model_path} for compatibility")
            
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
            # Load best model for export (use the standard name)
            best_model_path = Path("../models/military_target_detector.pt")
            if not best_model_path.exists():
                # Fall back to alternative name
                best_model_path = Path("../models/military_targets_best.pt")
            
            best_model = YOLO(str(best_model_path))
            
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
    # Config file approach (original)
    parser.add_argument('--config', default=None, help='Configuration file path')
    # YOLOv8 CLI arguments (for workflow.py compatibility)
    parser.add_argument('--data', help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--imgsz', type=int, help='Image size')
    parser.add_argument('--device', default=None, help='Device to use (cpu, mps, cuda)')
    # Export and validation
    parser.add_argument('--export', default='onnx', help='Export format (onnx, torchscript, etc.)')
    parser.add_argument('--validate', action='store_true', help='Validate model after training')
    
    args = parser.parse_args()
    
    # If CLI args provided, create temp config
    if args.data is not None:
        import tempfile
        import yaml
        
        # Load base config if it exists, otherwise use defaults
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default config (without data section - we'll use the dataset YAML directly)
            config = {
                'model': 'yolov8m.pt',
                'epochs': 100,
                'batch': 16,
                'imgsz': 640,
                'device': None,
                'project': 'runs/detect',
                'name': 'military_targets',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'auto',
                'verbose': True,
                'seed': 0,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            }
        
        # Override with CLI arguments (but keep data as just the path string)
        config['data'] = args.data  # This should be the path to dataset.yaml
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch:
            config['batch'] = args.batch
        if args.imgsz:
            config['imgsz'] = args.imgsz
        if args.device:
            config['device'] = args.device
            
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config = f.name
        
        config_path = temp_config
    else:
        # Use provided config file or default
        config_path = args.config if args.config else 'config.yaml'
    
    # Initialize trainer
    trainer = MilitaryTargetTrainer(config_path)
    
    # Train model
    logger.info("Starting military target detection training")
    results = trainer.train()
    
    # Clean up temp config if created
    if args.data is not None:
        try:
            os.unlink(temp_config)
        except:
            pass
    
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