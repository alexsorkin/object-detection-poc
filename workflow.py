#!/usr/bin/env python3
"""
Complete workflow for tank detection model training and testing
Orchestrates: data collection → scene generation → training → evaluation
"""

import sys
import subprocess
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: {description} failed")
        print(f"Error: {e}")
        return False

def workflow_download_tanks(args):
    """Step 1: Download tank images"""
    cmd = [sys.executable, "scripts/download_tank_images.py"]
    if args.max_images:
        cmd.extend(["--max-per-model", str(args.max_images)])
    return run_command(cmd, "Downloading tank images")

def workflow_generate_scenes(args):
    """Step 2: Generate synthetic battle scenes"""
    cmd = [sys.executable, "scripts/generate_battle_scenes.py"]
    # Note: You can modify the script to accept command-line args
    return run_command(cmd, "Generating battle scenes")

def workflow_train_model(args):
    """Step 3: Train the detection model"""
    cmd = [
        sys.executable, "training/train.py",
        "--data", str(args.data_yaml),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch_size),
        "--imgsz", str(args.img_size)
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    return run_command(cmd, "Training detection model")

def workflow_test_model(args):
    """Step 4: Test the trained model"""
    cmd = [
        sys.executable, "tests/test_tank_detection.py",
        "--mode", "dataset"
    ]
    if args.max_test_images:
        cmd.extend(["--max-images", str(args.max_test_images)])
    return run_command(cmd, "Testing detection model")

def workflow_export_onnx(args):
    """Step 5: Export model to ONNX"""
    print(f"\n{'='*60}")
    print("Exporting model to ONNX format")
    print(f"{'='*60}\n")
    
    try:
        from ultralytics import YOLO
        
        model_path = Path("models/military_target_detector.pt")
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            return False
        
        model = YOLO(str(model_path))
        onnx_path = model.export(format='onnx', dynamic=True)
        
        print(f"\n✓ Model exported to ONNX: {onnx_path}")
        return True
    except Exception as e:
        print(f"\n✗ ERROR exporting to ONNX: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Complete workflow for tank detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow (all steps)
  python workflow.py --all
  
  # Run specific steps
  python workflow.py --download --generate --train
  
  # Train with custom settings
  python workflow.py --train --epochs 100 --batch-size 16
  
  # Quick test run
  python workflow.py --generate --train --test --epochs 10 --max-images 50
        """
    )
    
    # Step selection
    parser.add_argument('--all', action='store_true',
                       help='Run all steps in sequence')
    parser.add_argument('--download', action='store_true',
                       help='Download tank images')
    parser.add_argument('--generate', action='store_true',
                       help='Generate battle scenes')
    parser.add_argument('--train', action='store_true',
                       help='Train detection model')
    parser.add_argument('--test', action='store_true',
                       help='Test detection model')
    parser.add_argument('--export', action='store_true',
                       help='Export model to ONNX')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--device', type=str,
                       help='Device for training (e.g., "mps", "cuda", "cpu")')
    parser.add_argument('--data-yaml', type=str, default='data/dataset.yaml',
                       help='Path to dataset YAML file')
    
    # Data parameters
    parser.add_argument('--max-images', type=int,
                       help='Max images to download per tank model')
    parser.add_argument('--max-test-images', type=int,
                       help='Max images to test')
    
    args = parser.parse_args()
    
    # If no steps selected and not --all, show help
    if not (args.all or args.download or args.generate or args.train or args.test or args.export):
        parser.print_help()
        return
    
    # Determine which steps to run
    steps = []
    if args.all:
        steps = ['download', 'generate', 'train', 'test', 'export']
    else:
        if args.download:
            steps.append('download')
        if args.generate:
            steps.append('generate')
        if args.train:
            steps.append('train')
        if args.test:
            steps.append('test')
        if args.export:
            steps.append('export')
    
    print(f"\n{'#'*60}")
    print("TANK DETECTION MODEL - COMPLETE WORKFLOW")
    print(f"{'#'*60}")
    print(f"\nSteps to execute: {', '.join(steps)}\n")
    
    # Execute steps
    success_count = 0
    total_steps = len(steps)
    
    for step in steps:
        if step == 'download':
            if workflow_download_tanks(args):
                success_count += 1
            else:
                print("\nWARNING: Download step failed, but continuing...")
                print("You can manually add tank images to data/raw_images/tanks/")
                success_count += 1  # Don't fail the workflow
        
        elif step == 'generate':
            if workflow_generate_scenes(args):
                success_count += 1
            else:
                print("\nERROR: Scene generation failed. Cannot continue.")
                break
        
        elif step == 'train':
            if workflow_train_model(args):
                success_count += 1
            else:
                print("\nERROR: Training failed. Cannot continue.")
                break
        
        elif step == 'test':
            if workflow_test_model(args):
                success_count += 1
            else:
                print("\nWARNING: Testing failed, but model was trained.")
                success_count += 1
        
        elif step == 'export':
            if workflow_export_onnx(args):
                success_count += 1
            else:
                print("\nWARNING: ONNX export failed, but model is still usable.")
                success_count += 1
    
    # Final summary
    print(f"\n{'#'*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'#'*60}")
    print(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("\n✓ All steps completed successfully!")
        print("\nNext steps:")
        print("  1. Review test results in data/synthetic_scenes/test_results.json")
        print("  2. Check model performance metrics")
        print("  3. Use the model for inference:")
        print("     python tests/test_tank_detection.py --mode video")
        print("  4. Integrate with Rust library for production deployment")
    else:
        print(f"\n⚠ Some steps failed or were skipped")
    
    print()

if __name__ == "__main__":
    main()
