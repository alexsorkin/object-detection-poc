#!/usr/bin/env python3
"""
Download images of Soviet/Russian tanks for training military target detection
Focuses on T-60s, T-70s, and T-80s series tanks
"""

import os
import requests
import time
from pathlib import Path
from typing import List, Dict
import json

# Tank series to download
TANK_SERIES = {
    "T-60": ["T-62", "T-64", "T-64A", "T-64B", "T-64BV"],
    "T-70": ["T-72", "T-72A", "T-72B", "T-72B3", "T-72M"],
    "T-80": ["T-80", "T-80B", "T-80BV", "T-80U", "T-80UK", "T-80UM"],
    "T-90": ["T-90", "T-90A", "T-90M"],  # Bonus: Modern variants
}

class TankImageDownloader:
    def __init__(self, output_dir: str = "../data/raw_images/tanks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "download_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load existing metadata if available"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"downloaded": {}, "stats": {}}
    
    def _save_metadata(self):
        """Save download metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def download_from_wikimedia(self, tank_model: str, max_images: int = 20):
        """
        Download tank images from Wikimedia Commons
        Note: This is a placeholder - actual implementation would use Wikimedia API
        """
        print(f"Downloading images for: {tank_model}")
        
        # Create directory for this tank model
        tank_dir = self.output_dir / tank_model.replace(" ", "_")
        tank_dir.mkdir(exist_ok=True)
        
        # Wikimedia Commons API endpoint
        api_url = "https://commons.wikimedia.org/w/api.php"
        
        # Search parameters
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{tank_model} tank",
            "gsrlimit": max_images,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": 1024,
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                if "query" in data and "pages" in data["query"]:
                    pages = data["query"]["pages"]
                    downloaded = 0
                    
                    for page_id, page in pages.items():
                        if "imageinfo" in page:
                            for img in page["imageinfo"]:
                                if "url" in img:
                                    img_url = img["url"]
                                    self._download_image(img_url, tank_dir, tank_model, downloaded)
                                    downloaded += 1
                                    time.sleep(0.5)  # Be nice to the server
                    
                    self.metadata["stats"][tank_model] = downloaded
                    print(f"  Downloaded {downloaded} images")
                else:
                    print(f"  No images found for {tank_model}")
                    
        except Exception as e:
            print(f"  Error downloading {tank_model}: {e}")
    
    def _download_image(self, url: str, output_dir: Path, tank_model: str, index: int):
        """Download a single image"""
        try:
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                # Determine file extension
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'
                elif 'png' in content_type:
                    ext = 'png'
                else:
                    ext = 'jpg'  # default
                
                filename = f"{tank_model.replace(' ', '_')}_{index:04d}.{ext}"
                filepath = output_dir / filename
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.metadata["downloaded"][str(filepath)] = {
                    "source": "wikimedia",
                    "url": url,
                    "tank_model": tank_model
                }
                print(f"    Downloaded: {filename}")
                
        except Exception as e:
            print(f"    Failed to download image: {e}")
    
    def download_sample_dataset(self):
        """Download sample images from public military vehicle datasets"""
        print("\n=== Downloading Sample Tank Images ===\n")
        
        # Sample URLs from public domain military archives
        # These are placeholder URLs - replace with actual dataset sources
        sample_urls = [
            # Add actual URLs here from:
            # - Military museums
            # - Defense industry websites
            # - Government archives
            # - Academic datasets
        ]
        
        print("Note: For actual image downloads, you should:")
        print("  1. Use military museum archives")
        print("  2. Access defense contractor public galleries")
        print("  3. Use government declassified image databases")
        print("  4. Comply with licensing and attribution requirements")
        print()
    
    def download_all_tanks(self, images_per_model: int = 20):
        """Download images for all tank models"""
        print("\n" + "="*60)
        print("TANK IMAGE DOWNLOADER")
        print("="*60 + "\n")
        
        total_models = sum(len(models) for models in TANK_SERIES.values())
        current = 0
        
        for series, models in TANK_SERIES.items():
            print(f"\n--- {series} Series ---")
            for model in models:
                current += 1
                print(f"[{current}/{total_models}] {model}")
                self.download_from_wikimedia(model, images_per_model)
        
        self._save_metadata()
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total models processed: {total_models}")
        print(f"Total images downloaded: {sum(self.metadata['stats'].values())}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Metadata saved: {self.metadata_file}")
        print()

def create_manual_download_list():
    """Generate a list of search queries for manual image collection"""
    output_file = Path("../data/tank_image_sources.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("MANUAL IMAGE COLLECTION GUIDE\n")
        f.write("="*60 + "\n\n")
        
        f.write("RECOMMENDED IMAGE SOURCES:\n")
        f.write("-"*60 + "\n")
        f.write("1. Wikimedia Commons: https://commons.wikimedia.org/\n")
        f.write("2. Military Museums (Kubinka Tank Museum, etc.)\n")
        f.write("3. Defense Ministry Archives (various countries)\n")
        f.write("4. Academic Datasets (COCO, ImageNet subsets)\n")
        f.write("5. News Archives (Reuters, AP, etc.)\n\n")
        
        f.write("SEARCH QUERIES BY TANK MODEL:\n")
        f.write("-"*60 + "\n\n")
        
        for series, models in TANK_SERIES.items():
            f.write(f"\n{series} Series:\n")
            for model in models:
                f.write(f"  - \"{model} tank\"\n")
                f.write(f"  - \"{model} military vehicle\"\n")
                f.write(f"  - \"{model} main battle tank\"\n")
        
        f.write("\n\nIMAGE REQUIREMENTS:\n")
        f.write("-"*60 + "\n")
        f.write("- Minimum resolution: 640x640 pixels\n")
        f.write("- Format: JPG or PNG\n")
        f.write("- Various angles: front, side, rear, 3/4 view\n")
        f.write("- Different conditions: day, dusk, winter, muddy\n")
        f.write("- Different distances: close-up, medium, far\n")
        f.write("- License: Public domain or permissive license\n")
        
    print(f"\nManual download guide created: {output_file}")

if __name__ == "__main__":
    print("Tank Image Downloader for Military Target Detection")
    print("="*60)
    print()
    print("IMPORTANT NOTES:")
    print("- This script provides a framework for downloading tank images")
    print("- Actual downloads require appropriate API keys and permissions")
    print("- Always respect copyright and licensing requirements")
    print("- For training data, ensure diverse angles and conditions")
    print()
    
    # Create manual download guide
    create_manual_download_list()
    
    # Initialize downloader
    downloader = TankImageDownloader()
    
    print("\nREADY TO DOWNLOAD")
    print("-"*60)
    print("To proceed with automated download:")
    print("  1. Ensure you have API credentials for image sources")
    print("  2. Review and accept terms of service for each source")
    print("  3. Uncomment the download line below")
    print()
    
    # Uncomment to start downloading:
    # downloader.download_all_tanks(images_per_model=20)
    
    print("Setup complete. Review tank_image_sources.txt for manual collection guide.")
