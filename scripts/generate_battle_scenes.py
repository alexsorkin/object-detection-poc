#!/usr/bin/env python3
"""
Generate synthetic battle scenes with tanks for testing detection models
Creates realistic composite images with tanks in various battlefield scenarios
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import random

class BattleSceneGenerator:
    def __init__(self, 
                 tank_images_dir: str = "data/raw_images/tanks",
                 backgrounds_dir: str = "data/backgrounds",
                 output_dir: str = "data/synthetic_scenes"):
        self.tank_images_dir = Path(tank_images_dir)
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        self.tank_images = []
        self.background_images = []
        
    def load_assets(self):
        """Load tank and background images"""
        print("Loading assets...")
        
        # Load tank images (just collect paths, validate on use)
        if self.tank_images_dir.exists():
            for img_path in self.tank_images_dir.rglob("*.jpg"):
                if img_path.stat().st_size > 0:  # Quick check: file not empty
                    self.tank_images.append(img_path)
            
            for img_path in self.tank_images_dir.rglob("*.png"):
                if img_path.stat().st_size > 0:
                    self.tank_images.append(img_path)
        
        # Load or generate background images
        if self.backgrounds_dir.exists():
            for img_path in self.backgrounds_dir.glob("*.jpg"):
                self.background_images.append(img_path)
            for img_path in self.backgrounds_dir.glob("*.png"):
                self.background_images.append(img_path)
        
        print(f"  Loaded {len(self.tank_images)} tank images")
        print(f"  Loaded {len(self.background_images)} background images")
        
        if len(self.background_images) == 0:
            print("  Generating synthetic backgrounds...")
            self._generate_backgrounds(count=50)
    
    def _generate_backgrounds(self, count: int = 50):
        """Generate synthetic battlefield backgrounds"""
        self.backgrounds_dir.mkdir(parents=True, exist_ok=True)
        
        background_types = [
            self._generate_field_background,
            self._generate_urban_background,
            self._generate_forest_background,
            self._generate_desert_background,
            self._generate_mountain_background,
        ]
        
        for i in range(count):
            bg_type = random.choice(background_types)
            bg = bg_type(1920, 1080)
            
            bg_path = self.backgrounds_dir / f"synthetic_bg_{i:04d}.jpg"
            cv2.imwrite(str(bg_path), bg)
            self.background_images.append(bg_path)
        
        print(f"  Generated {count} synthetic backgrounds")
    
    def _generate_synthetic_tank(self, width: int = 200, height: int = 100) -> Image.Image:
        """Generate a synthetic tank-like shape when no real images available"""
        # Create tank silhouette
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Tank body (rectangle)
        body_color = (random.randint(60, 100), random.randint(60, 100), random.randint(60, 100), 255)
        draw.rectangle([20, 40, 180, 90], fill=body_color, outline=(40, 40, 40, 255))
        
        # Turret (ellipse/rectangle)
        turret_color = tuple(max(0, c - 20) for c in body_color[:3]) + (255,)
        draw.ellipse([60, 20, 140, 60], fill=turret_color, outline=(40, 40, 40, 255))
        
        # Barrel (rectangle)
        barrel_length = random.randint(60, 90)
        draw.rectangle([140, 35, 140 + barrel_length, 45], fill=turret_color, outline=(40, 40, 40, 255))
        
        # Tracks/wheels (circles)
        for x in [40, 70, 100, 130, 160]:
            draw.ellipse([x-8, 75, x+8, 95], fill=(30, 30, 30, 255), outline=(20, 20, 20, 255))
        
        return img
    
    def _generate_field_background(self, width: int, height: int) -> np.ndarray:
        """Generate grassy field background"""
        # Base green color with variation
        base_color = np.array([80, 120, 50])  # BGR
        variation = np.random.randint(-30, 30, (height, width, 3))
        bg = np.clip(base_color + variation, 0, 255).astype(np.uint8)
        
        # Add texture
        noise = np.random.randint(-20, 20, (height, width, 3))
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
        
        # Add sky
        sky_color = np.array([200, 150, 100])  # Light blue-gray
        for y in range(height // 3):
            alpha = y / (height // 3)
            bg[y] = (1 - alpha) * sky_color + alpha * bg[y]
        
        return bg
    
    def _generate_urban_background(self, width: int, height: int) -> np.ndarray:
        """Generate urban/destroyed city background"""
        # Gray/brown base
        base_color = np.array([100, 100, 110])
        variation = np.random.randint(-40, 40, (height, width, 3))
        bg = np.clip(base_color + variation, 0, 255).astype(np.uint8)
        
        # Add building-like structures
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, width - 200)
            w = random.randint(100, 300)
            h = random.randint(200, 500)
            y = height - h
            color = (random.randint(60, 100), random.randint(60, 100), random.randint(70, 110))
            cv2.rectangle(bg, (x, y), (x + w, height), color, -1)
        
        return bg
    
    def _generate_forest_background(self, width: int, height: int) -> np.ndarray:
        """Generate forest background"""
        # Dark green base
        base_color = np.array([30, 80, 30])
        variation = np.random.randint(-20, 40, (height, width, 3))
        bg = np.clip(base_color + variation, 0, 255).astype(np.uint8)
        
        # Add tree-like shapes
        for _ in range(random.randint(20, 40)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(20, 60)
            color = (random.randint(20, 50), random.randint(60, 100), random.randint(20, 50))
            cv2.circle(bg, (x, y), radius, color, -1)
        
        return bg
    
    def _generate_desert_background(self, width: int, height: int) -> np.ndarray:
        """Generate desert background"""
        # Sand color
        base_color = np.array([100, 170, 200])  # BGR
        variation = np.random.randint(-30, 30, (height, width, 3))
        bg = np.clip(base_color + variation, 0, 255).astype(np.uint8)
        
        # Add dunes
        for _ in range(random.randint(5, 10)):
            x = random.randint(-200, width + 200)
            y = random.randint(height // 2, height)
            radius = random.randint(200, 500)
            color = tuple(int(c * random.uniform(0.9, 1.1)) for c in base_color)
            cv2.circle(bg, (x, y), radius, color, -1)
        
        return bg
    
    def _generate_mountain_background(self, width: int, height: int) -> np.ndarray:
        """Generate mountainous background"""
        # Gray/brown mountains
        base_color = np.array([90, 110, 100])
        bg = np.full((height, width, 3), base_color, dtype=np.uint8)
        
        # Add mountain silhouettes
        points = []
        for x in range(0, width, 50):
            y = random.randint(height // 4, height // 2)
            points.append([x, y])
        points.append([width, height])
        points.append([0, height])
        
        mountain_color = (random.randint(60, 90), random.randint(70, 100), random.randint(60, 90))
        cv2.fillPoly(bg, [np.array(points)], mountain_color)
        
        return bg
    
    def generate_scene(self, 
                      scene_id: int,
                      num_tanks: int = None,
                      background_path: str = None) -> Dict:
        """Generate a single battle scene with tanks"""
        
        if num_tanks is None:
            num_tanks = random.randint(1, 5)
        
        # Load or create background
        if background_path is None and self.background_images:
            background_path = random.choice(self.background_images)
        
        if background_path:
            bg = cv2.imread(str(background_path))
            if bg is None:
                bg = self._generate_field_background(1920, 1080)
        else:
            bg = self._generate_field_background(1920, 1080)
        
        height, width = bg.shape[:2]
        
        # Convert to PIL for easier compositing
        bg_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        
        annotations = []
        
        # Place tanks in the scene
        for i in range(num_tanks):
            # Use real tank image if available, otherwise generate synthetic
            if self.tank_images:
                tank_path = random.choice(self.tank_images)
                try:
                    tank_img = Image.open(tank_path).convert('RGBA')
                    source_name = str(tank_path.name)
                except Exception as e:
                    # If image is corrupted, skip it or use synthetic
                    print(f"  Warning: Could not load {tank_path.name}: {e}")
                    tank_img = self._generate_synthetic_tank(
                        width=random.randint(150, 250),
                        height=random.randint(75, 125)
                    )
                    source_name = "synthetic_tank"
            else:
                # Generate synthetic tank shape
                tank_img = self._generate_synthetic_tank(
                    width=random.randint(150, 250),
                    height=random.randint(75, 125)
                )
                source_name = "synthetic_tank"
            
            # Random tank size (simulate distance)
            scale = random.uniform(0.1, 0.5)
            new_size = (int(tank_img.width * scale), int(tank_img.height * scale))
            tank_img = tank_img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply transformations BEFORE positioning
            # Rotation (expand=True changes size, so do this first)
            angle = random.uniform(-30, 30)
            tank_img = tank_img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
            
            # Lighting adjustments
            enhancer = ImageEnhance.Brightness(tank_img)
            tank_img = enhancer.enhance(random.uniform(0.7, 1.3))
            
            # Now calculate position with final tank size
            max_x = width - tank_img.width - 50
            max_y = height - tank_img.height - 50
            
            # Skip if tank is too large for the scene
            if max_x < 50 or max_y < 50:
                continue
            
            # Ensure valid y range (tanks on ground, but check bounds)
            min_y = max(50, height // 3)
            if min_y > max_y:
                # Tank is too large, skip it
                continue
            
            x = random.randint(50, max_x)
            y = random.randint(min_y, max_y)
            
            # Paste tank onto background
            bg_pil.paste(tank_img, (x, y), tank_img)
            
            # Create YOLO format annotation
            tank_center_x = (x + tank_img.width / 2) / width
            tank_center_y = (y + tank_img.height / 2) / height
            tank_width = tank_img.width / width
            tank_height = tank_img.height / height
            
            annotations.append({
                "class": 2,  # military_vehicle class
                "class_name": "military_vehicle",
                "bbox": [tank_center_x, tank_center_y, tank_width, tank_height],
                "bbox_pixels": [x, y, tank_img.width, tank_img.height],
                "source_image": source_name
            })
        
        # Convert back to OpenCV format
        final_scene = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
        
        # Save scene
        scene_filename = f"battle_scene_{scene_id:05d}.jpg"
        scene_path = self.output_dir / "images" / scene_filename
        cv2.imwrite(str(scene_path), final_scene)
        
        # Save annotations in YOLO format
        label_filename = f"battle_scene_{scene_id:05d}.txt"
        label_path = self.output_dir / "labels" / label_filename
        with open(label_path, 'w') as f:
            for ann in annotations:
                # YOLO format: class x_center y_center width height
                f.write(f"{ann['class']} {ann['bbox'][0]:.6f} {ann['bbox'][1]:.6f} "
                       f"{ann['bbox'][2]:.6f} {ann['bbox'][3]:.6f}\n")
        
        # Save visualization
        vis_scene = final_scene.copy()
        for ann in annotations:
            x, y, w, h = ann['bbox_pixels']
            cv2.rectangle(vis_scene, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(vis_scene, ann['class_name'], (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        vis_filename = f"battle_scene_{scene_id:05d}_vis.jpg"
        vis_path = self.output_dir / "visualizations" / vis_filename
        cv2.imwrite(str(vis_path), vis_scene)
        
        return {
            "scene_id": scene_id,
            "image_path": str(scene_path),
            "label_path": str(label_path),
            "num_tanks": len(annotations),
            "annotations": annotations
        }
    
    def generate_dataset(self, num_scenes: int = 100):
        """Generate a complete dataset of battle scenes"""
        print(f"\n{'='*60}")
        print("BATTLE SCENE GENERATOR")
        print(f"{'='*60}\n")
        
        self.load_assets()
        
        if not self.tank_images:
            print("\n" + "!"*60)
            print("NOTE: No real tank images found!")
            print("Generating scenes with SYNTHETIC tank shapes.")
            print("For better results, add real tank images to:")
            print(f"  {self.tank_images_dir.absolute()}")
            print("Or run: python scripts/download_tank_images.py")
            print("!"*60 + "\n")
        
        metadata = []
        
        print(f"\nGenerating {num_scenes} battle scenes...")
        for i in range(num_scenes):
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_scenes} scenes")
            
            scene_data = self.generate_scene(i)
            metadata.append(scene_data)
        
        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total scenes generated: {num_scenes}")
        print(f"Total tanks placed: {sum(s['num_tanks'] for s in metadata)}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"  - Images: {self.output_dir / 'images'}")
        print(f"  - Labels: {self.output_dir / 'labels'}")
        print(f"  - Visualizations: {self.output_dir / 'visualizations'}")
        print(f"  - Metadata: {metadata_path}")
        print()

if __name__ == "__main__":
    generator = BattleSceneGenerator()
    generator.generate_dataset(num_scenes=100)
