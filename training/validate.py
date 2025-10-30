#!/usr/bin/env python3
"""
Model validation and testing utilities for military target detection
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class ModelValidator:
    """Validation utilities for military target detection model"""
    
    def __init__(self, model_path: str):
        """Initialize validator with trained model"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.class_names = ['armed_personnel', 'rocket_launcher', 'military_vehicle', 'heavy_weapon']
        
    def validate_on_video(self, video_path: str, output_path: str = None, conf_threshold: float = 0.5):
        """
        Validate model performance on video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            conf_threshold: Confidence threshold for detections
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections = []
        processing_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Run inference
            start_time = cv2.getTickCount()
            results = self.model(frame, conf=conf_threshold, verbose=False)
            end_time = cv2.getTickCount()
            
            # Calculate processing time
            processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # ms
            processing_times.append(processing_time)
            
            # Extract detections
            frame_detections = []
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    frame_detections.append({
                        'frame': frame_count,
                        'class': self.class_names[cls_id],
                        'confidence': float(conf),
                        'bbox': box.tolist()
                    })
                    
                    # Draw bounding box
                    if output_path:
                        x1, y1, x2, y2 = box.astype(int)
                        color = self._get_class_color(cls_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{self.class_names[cls_id]}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.extend(frame_detections)
            
            # Add FPS info
            if output_path:
                fps_text = f"FPS: {1000/processing_time:.1f} | Time: {processing_time:.1f}ms"
                cv2.putText(frame, fps_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(frame)
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            writer.release()
            
        # Calculate statistics
        avg_processing_time = np.mean(processing_times)
        avg_fps = 1000 / avg_processing_time
        
        logger.info(f"Video validation completed:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Total detections: {len(detections)}")
        logger.info(f"  Average processing time: {avg_processing_time:.2f}ms")
        logger.info(f"  Average FPS: {avg_fps:.1f}")
        
        return {
            'total_frames': frame_count,
            'total_detections': len(detections),
            'detections': detections,
            'avg_processing_time_ms': avg_processing_time,
            'avg_fps': avg_fps,
            'processing_times': processing_times
        }
    
    def benchmark_inference_speed(self, image_size: Tuple[int, int] = (640, 640), num_iterations: int = 100):
        """
        Benchmark model inference speed
        
        Args:
            image_size: Input image size for benchmarking
            num_iterations: Number of inference iterations
        """
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            self.model(dummy_image, verbose=False)
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = cv2.getTickCount()
            results = self.model(dummy_image, verbose=False)
            end_time = cv2.getTickCount()
            
            processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # ms
            times.append(processing_time)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_iterations} iterations")
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        avg_fps = 1000 / avg_time
        
        logger.info(f"Inference speed benchmark:")
        logger.info(f"  Image size: {image_size}")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Average time: {avg_time:.2f} ± {std_time:.2f}ms")
        logger.info(f"  Min time: {min_time:.2f}ms")
        logger.info(f"  Max time: {max_time:.2f}ms")
        logger.info(f"  Average FPS: {avg_fps:.1f}")
        
        return {
            'image_size': image_size,
            'iterations': num_iterations,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'avg_fps': avg_fps,
            'times': times
        }
    
    def analyze_class_distribution(self, detections: List[Dict]) -> Dict:
        """Analyze class distribution in detections"""
        class_counts = {cls: 0 for cls in self.class_names}
        confidence_by_class = {cls: [] for cls in self.class_names}
        
        for detection in detections:
            cls_name = detection['class']
            conf = detection['confidence']
            
            class_counts[cls_name] += 1
            confidence_by_class[cls_name].append(conf)
        
        # Calculate statistics
        stats = {}
        for cls_name in self.class_names:
            confidences = confidence_by_class[cls_name]
            stats[cls_name] = {
                'count': class_counts[cls_name],
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'std_confidence': np.std(confidences) if confidences else 0.0,
                'min_confidence': np.min(confidences) if confidences else 0.0,
                'max_confidence': np.max(confidences) if confidences else 0.0
            }
        
        return stats
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class visualization"""
        colors = [
            (255, 0, 0),    # Red - armed_personnel
            (0, 255, 0),    # Green - rocket_launcher  
            (0, 0, 255),    # Blue - military_vehicle
            (255, 255, 0)   # Yellow - heavy_weapon
        ]
        return colors[class_id % len(colors)]
    
    def create_performance_report(self, results: Dict, output_path: str):
        """Create detailed performance report"""
        report = {
            'model_path': self.model_path,
            'class_names': self.class_names,
            'performance_metrics': results,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {output_path}")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate military target detection model')
    parser.add_argument('model', help='Path to trained model')
    parser.add_argument('--video', help='Path to validation video')
    parser.add_argument('--output', help='Path to save annotated video')
    parser.add_argument('--benchmark', action='store_true', help='Run inference speed benchmark')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--report', help='Path to save performance report')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ModelValidator(args.model)
    
    results = {}
    
    # Video validation
    if args.video:
        logger.info(f"Validating on video: {args.video}")
        video_results = validator.validate_on_video(args.video, args.output, args.conf)
        results['video_validation'] = video_results
        
        # Analyze detections
        class_stats = validator.analyze_class_distribution(video_results['detections'])
        results['class_distribution'] = class_stats
    
    # Speed benchmark
    if args.benchmark:
        logger.info("Running inference speed benchmark")
        benchmark_results = validator.benchmark_inference_speed()
        results['speed_benchmark'] = benchmark_results
    
    # Save report
    if args.report:
        validator.create_performance_report(results, args.report)
    
    logger.info("Validation completed")

if __name__ == "__main__":
    main()