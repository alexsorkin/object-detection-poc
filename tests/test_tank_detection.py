#!/usr/bin/env python3
"""
Integration tests for tank detection pipeline
Tests the complete flow: training → inference → evaluation
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
from ultralytics import YOLO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TankDetectionTester:
    def __init__(self, 
                 model_path: str = "../models/military_target_detector.pt",
                 test_scenes_dir: str = "../data/synthetic_scenes"):
        self.model_path = Path(model_path)
        self.test_scenes_dir = Path(test_scenes_dir)
        self.model = None
        
    def load_model(self):
        """Load the trained YOLO model"""
        print(f"Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            print(f"ERROR: Model not found at {self.model_path}")
            print("Please train the model first using training/train.py")
            return False
        
        try:
            self.model = YOLO(str(self.model_path))
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False
    
    def test_single_image(self, image_path: Path, show_viz: bool = True):
        """Test detection on a single image"""
        if self.model is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return None
        
        print(f"\nTesting: {image_path.name}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  ERROR: Could not load image")
            return None
        
        # Run inference
        results = self.model(img, conf=0.25)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = result.names[cls]
                
                detections.append({
                    "class": cls,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
        
        print(f"  Detected {len(detections)} objects")
        for det in detections:
            print(f"    - {det['class_name']}: {det['confidence']:.2f}")
        
        if show_viz and detections:
            # Draw detections
            vis_img = img.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                color = (0, 255, 0)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(vis_img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Save visualization
            vis_dir = self.test_scenes_dir / "test_results"
            vis_dir.mkdir(exist_ok=True)
            vis_path = vis_dir / f"{image_path.stem}_detection.jpg"
            cv2.imwrite(str(vis_path), vis_img)
            print(f"  Visualization saved: {vis_path}")
        
        return detections
    
    def test_dataset(self, max_images: int = None):
        """Test detection on the entire synthetic dataset"""
        print(f"\n{'='*60}")
        print("TANK DETECTION TEST SUITE")
        print(f"{'='*60}\n")
        
        # Load model
        if not self.load_model():
            return
        
        # Find test images
        images_dir = self.test_scenes_dir / "images"
        labels_dir = self.test_scenes_dir / "labels"
        
        if not images_dir.exists():
            print(f"ERROR: Test images not found at {images_dir}")
            print("Please generate battle scenes first using scripts/generate_battle_scenes.py")
            return
        
        image_paths = sorted(images_dir.glob("*.jpg"))
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Testing on {len(image_paths)} images\n")
        
        # Load metadata if available
        metadata_path = self.test_scenes_dir / "dataset_metadata.json"
        ground_truth = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                for item in metadata:
                    scene_id = item['scene_id']
                    ground_truth[scene_id] = item['annotations']
        
        # Run tests
        results = []
        total_tp = 0  # True positives
        total_fp = 0  # False positives
        total_fn = 0  # False negatives
        
        for img_path in image_paths:
            detections = self.test_single_image(img_path, show_viz=False)
            if detections is None:
                continue
            
            # Extract scene ID from filename
            scene_id = int(img_path.stem.split('_')[-1])
            
            # Compare with ground truth if available
            if scene_id in ground_truth:
                gt_count = len(ground_truth[scene_id])
                det_count = len(detections)
                
                # Simple counting-based metrics (simplified)
                tp = min(gt_count, det_count)
                fp = max(0, det_count - gt_count)
                fn = max(0, gt_count - det_count)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            results.append({
                "image": img_path.name,
                "scene_id": scene_id,
                "num_detections": len(detections),
                "detections": detections
            })
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"Images tested: {len(results)}")
        print(f"Total detections: {sum(r['num_detections'] for r in results)}")
        
        if ground_truth:
            print(f"\nDetection Metrics:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1:.3f}")
        
        # Save results
        results_path = self.test_scenes_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "summary": {
                    "num_images": len(results),
                    "total_detections": sum(r['num_detections'] for r in results),
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                },
                "results": results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        print()
    
    def test_video_stream(self, video_path: str = None, show_live: bool = True):
        """Test real-time detection on video stream"""
        if self.model is None:
            print("ERROR: Model not loaded. Call load_model() first.")
            return
        
        print("\n=== LIVE VIDEO DETECTION TEST ===\n")
        
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"Testing on video: {video_path}")
        else:
            cap = cv2.VideoCapture(0)
            print("Testing on webcam (press 'q' to quit)")
        
        frame_count = 0
        fps_sum = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            import time
            start = time.time()
            results = self.model(frame, conf=0.25)
            inference_time = time.time() - start
            fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_sum += fps
            frame_count += 1
            
            # Draw results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = result.names[cls]
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if show_live:
                cv2.imshow('Tank Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        avg_fps = fps_sum / frame_count if frame_count > 0 else 0
        print(f"\nProcessed {frame_count} frames")
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test tank detection model')
    parser.add_argument('--model', type=str, default='../models/military_target_detector.pt',
                       help='Path to trained model')
    parser.add_argument('--scenes', type=str, default='../data/synthetic_scenes',
                       help='Path to test scenes directory')
    parser.add_argument('--mode', type=str, choices=['single', 'dataset', 'video'],
                       default='dataset', help='Test mode')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--video', type=str, help='Path to video file (for video mode)')
    parser.add_argument('--max-images', type=int, help='Max images to test (for dataset mode)')
    
    args = parser.parse_args()
    
    tester = TankDetectionTester(
        model_path=args.model,
        test_scenes_dir=args.scenes
    )
    
    if args.mode == 'single':
        if not args.image:
            print("ERROR: --image required for single mode")
            sys.exit(1)
        tester.load_model()
        tester.test_single_image(Path(args.image))
    
    elif args.mode == 'dataset':
        tester.test_dataset(max_images=args.max_images)
    
    elif args.mode == 'video':
        tester.load_model()
        tester.test_video_stream(video_path=args.video)
