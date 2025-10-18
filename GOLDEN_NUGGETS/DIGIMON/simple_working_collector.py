#!/usr/bin/env python3
"""
Simple Working Digimon Collector
Based on successful test - gets the main Digimon image from each page
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin
from pathlib import Path
from PIL import Image, ImageOps
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = 'https://wikimon.net'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Expanded Digimon lists for comprehensive collection
DIGIMON_BY_STAGE = {
    'fresh': ['Koromon', 'Tsunomon', 'Yokomon', 'Tokomon', 'Punimon', 'Yuramon', 'Botamon', 'Poyomon', 'Nyaromon', 'Petitmon', 'Zerimon', 'Conomon', 'Cocomon', 'Kuramon', 'Pabumon'],
    
    'in_training': ['Agumon', 'Gabumon', 'Biyomon', 'Tentomon', 'Gomamon', 'Patamon', 'Palmon', 'Veemon', 'Wormmon', 'Hawkmon', 'Armadillomon', 'Guilmon', 'Renamon', 'Terriermon', 'Lopmon', 'Impmon', 'Monodramon', 'Dorumon', 'Kotemon', 'Muchomon'],
    
    'rookie': ['Greymon', 'Garurumon', 'Birdramon', 'Kabuterimon', 'Ikkakumon', 'Angemon', 'Togemon', 'Devimon', 'Bakemon', 'Meramon', 'Shellmon', 'Numemon', 'Monzaemon', 'Centarumon', 'Leomon', 'Ogremon', 'Elecmon', 'Frigimon', 'Mojyamon', 'Sukamon'],
    
    'champion': ['MetalGreymon', 'WereGarurumon', 'Garudamon', 'MegaKabuterimon', 'Zudomon', 'MagnaAngemon', 'Lillymon', 'SkullGreymon', 'MetalMamemon', 'Vademon', 'Digitamamon', 'Vegiemon', 'Pumpkinmon', 'Gotsumon', 'Gekomon', 'ShogunGekomon', 'Whamon', 'Kokatorimon', 'Tyrannomon', 'Machinedramon'],
    
    'ultimate': ['WarGreymon', 'MetalGarurumon', 'Phoenixmon', 'HerculesKabuterimon', 'Vikemon', 'Seraphimon', 'Rosemon', 'SkullMeramon', 'Boltmon', 'Myotismon', 'VenomMyotismon', 'Piedmon', 'MetalSeadramon', 'Puppetmon', 'Machinedramon', 'Apocalymon', 'Mugendramon', 'Pinochimon', 'MetalEtemon', 'SaberLeomon'],
    
    'mega': ['Omegamon', 'Alphamon', 'Imperialdramon', 'Dukemon', 'Gallantmon', 'MegaGargomon', 'Sakuyamon', 'Justimon', 'Leviamon', 'Daemon', 'Beelzemon', 'Cherubimon', 'Ophanimon', 'Lucemon', 'Susanoomon', 'AncientGreymon', 'AncientGarurumon', 'Examon', 'UlforceVeedramon', 'Craniummon']
}

STAGE_TARGETS = {
    'fresh': 12,
    'in_training': 15, 
    'rookie': 15,
    'champion': 15,
    'ultimate': 15,
    'mega': 12
}

class SimpleWorkingCollector:
    def __init__(self, output_dir: str = "simple_digimon_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        for stage in STAGE_TARGETS.keys():
            (self.output_dir / stage).mkdir(exist_ok=True)
            
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.stage_counts = {stage: 0 for stage in STAGE_TARGETS.keys()}
        self.successful = 0

    def find_main_digimon_image(self, soup, digimon_name: str) -> str:
        """Find the main Digimon image - simplified approach"""
        
        # Strategy 1: Look for the main image with the Digimon's name
        expected_filename = f"{digimon_name}.jpg"
        
        for img in soup.find_all('img'):
            src = img.get('src')
            if src and expected_filename.lower() in src.lower():
                # Found the main image
                logger.info(f"Found main image: {src}")
                
                # Convert thumbnail to full size if needed
                if '/thumb/' in src:
                    # Convert from /images/thumb/a/bc/Filename.jpg/320px-Filename.jpg
                    # to /images/a/bc/Filename.jpg
                    parts = src.split('/thumb/')
                    if len(parts) == 2:
                        path_part = parts[1]
                        path_components = path_part.split('/')
                        if len(path_components) >= 3:
                            hash_path = '/'.join(path_components[:2])
                            filename = path_components[2]
                            full_url = f"/images/{hash_path}/{filename}"
                            logger.info(f"Converted to full size: {full_url}")
                            return urljoin(BASE_URL, full_url)
                
                return urljoin(BASE_URL, src)
        
        # Strategy 2: Look for any large-ish image
        for img in soup.find_all('img'):
            src = img.get('src')
            if src and any(ext in src.lower() for ext in ['.jpg', '.png']):
                # Skip obvious icons and small images
                if not any(skip in src.lower() for skip in ['icon', 'logo', '15px', '20px', '30px']):
                    # Check if it's in a reasonable location
                    if '/images/' in src:
                        logger.info(f"Found fallback image: {src}")
                        return urljoin(BASE_URL, src)
        
        return None

    def get_digimon_type(self, digimon_name: str) -> str:
        """Simple type classification"""
        name_lower = digimon_name.lower()
        
        if 'dramon' in name_lower:
            return 'dragon'
        elif any(term in name_lower for term in ['metal', 'machine', 'gear']):
            return 'machine'
        elif any(term in name_lower for term in ['angel', 'seraph']):
            return 'holy'
        elif 'demon' in name_lower or 'devil' in name_lower:
            return 'dark'
        else:
            return 'beast'

    def process_digimon(self, digimon_name: str, evolution_stage: str) -> bool:
        """Process a single Digimon"""
        try:
            # Direct URL approach
            url = f"{BASE_URL}/{digimon_name}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find main image
            image_url = self.find_main_digimon_image(soup, digimon_name)
            if not image_url:
                logger.warning(f"No image found for {digimon_name}")
                return False
            
            # Download image
            img_response = self.session.get(image_url, timeout=30)
            img_response.raise_for_status()
            
            if len(img_response.content) < 5000:
                logger.warning(f"Image too small for {digimon_name}")
                return False
            
            # Process image
            image_data = BytesIO(img_response.content)
            
            with Image.open(image_data) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Crop white space
                bbox = self._get_content_bbox(img)
                if bbox:
                    img = img.crop(bbox)
                
                # Add padding
                padding = max(img.width, img.height) // 20
                img = ImageOps.expand(img, padding, fill='white')
                
                # Resize to 512x512
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                
                final_img = Image.new('RGB', (512, 512), 'white')
                paste_x = (512 - img.width) // 2
                paste_y = (512 - img.height) // 2
                final_img.paste(img, (paste_x, paste_y))
                
                # Save files
                clean_name = re.sub(r'[<>:"/\\|?*]', '_', digimon_name)
                stage_dir = self.output_dir / evolution_stage
                
                image_path = stage_dir / f"{clean_name}.png"
                caption_path = stage_dir / f"{clean_name}.txt"
                
                final_img.save(image_path, 'PNG', quality=95)
                
                # Create caption
                digimon_type = self.get_digimon_type(digimon_name)
                caption = f"a {clean_name} digimon, {evolution_stage} level, {digimon_type} type, in standard digimon art style, high quality digital artwork, clean background"
                
                with open(caption_path, 'w') as f:
                    f.write(caption)
                
                self.stage_counts[evolution_stage] += 1
                logger.info(f"✅ Saved {digimon_name} ({evolution_stage}) - {self.stage_counts[evolution_stage]}/{STAGE_TARGETS[evolution_stage]}")
                return True
                
        except Exception as e:
            logger.error(f"Error processing {digimon_name}: {e}")
            return False

    def _get_content_bbox(self, img):
        """Find content bounding box"""
        gray = img.convert('L')
        width, height = gray.size
        left, top, right, bottom = width, height, 0, 0
        
        pixels = gray.load()
        found_content = False
        
        for y in range(height):
            for x in range(width):
                if pixels[x, y] < 240:
                    found_content = True
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
        
        return (left, top, right + 1, bottom + 1) if found_content else None

    def collect_dataset(self):
        """Main collection process"""
        logger.info("Starting simple Digimon dataset collection...")
        
        total_attempted = 0
        
        for stage, digimon_list in DIGIMON_BY_STAGE.items():
            target = STAGE_TARGETS[stage]
            logger.info(f"\n🎯 Collecting {stage} Digimon (Target: {target})")
            
            for digimon_name in digimon_list:
                if self.stage_counts[stage] >= target:
                    break
                
                total_attempted += 1
                logger.info(f"Processing {digimon_name}...")
                
                if self.process_digimon(digimon_name, stage):
                    self.successful += 1
                
                time.sleep(2)  # Be polite
        
        # Report
        logger.info(f"\n🎉 Collection complete!")
        logger.info(f"📊 Success: {self.successful}/{total_attempted}")
        
        for stage, count in self.stage_counts.items():
            target = STAGE_TARGETS[stage]
            logger.info(f"  {stage}: {count}/{target} ({(count/target)*100:.1f}%)")

def main():
    collector = SimpleWorkingCollector()
    collector.collect_dataset()

if __name__ == "__main__":
    main()