#!/usr/bin/env python3
"""
Debug version of the fixed scraper to see exactly what's happening
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

# Better headers to avoid detection
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def find_best_image_debug(soup, digimon_name):
    """Find the best quality image on the page, with detailed debugging."""
    print(f"  === DEBUG: Finding best image for {digimon_name} ===")
    
    # Strategy 1: Look for the main infobox image
    print("  Strategy 1: Looking for infobox image...")
    image_link = soup.find('a', class_='image')
    if image_link and image_link.find('img'):
        img_tag = image_link.find('img')
        src = img_tag.get('src')
        print(f"    Found image link: {src}")
        if src and not is_placeholder(src):
            full_url = urljoin("https://wikimon.net", src)
            print(f"    ✅ Using infobox image: {full_url}")
            return full_url
        else:
            print(f"    ❌ Infobox image is placeholder or invalid")
    else:
        print("    No infobox image found")
    
    # Strategy 2: Look for images in the infobox table
    print("  Strategy 2: Looking for infobox table images...")
    infobox = soup.find('table', class_='infobox')
    if infobox:
        print("    Found infobox table")
        infobox_images = infobox.find_all('img')
        for i, img in enumerate(infobox_images):
            src = img.get('src')
            if src:
                full_url = urljoin("https://wikimon.net", src)
                print(f"      Image {i+1}: {full_url}")
                if not is_placeholder(src) and is_valid_image_url(src):
                    print(f"      ✅ Using infobox table image: {full_url}")
                    return full_url
    else:
        print("    No infobox table found")
    
    # Strategy 3: Look for any image that's not a placeholder
    print("  Strategy 3: Looking for any valid image...")
    all_images = soup.find_all('img')
    print(f"    Found {len(all_images)} total images")
    
    # Sort by potential quality (larger images first)
    valid_images = []
    for img in all_images:
        src = img.get('src')
        if src and not is_placeholder(src) and is_valid_image_url(src):
            width = img.get('width')
            height = img.get('height')
            if width and height:
                try:
                    w, h = int(width), int(height)
                    if w > 50 and h > 50:  # Avoid tiny images
                        valid_images.append((w * h, src, w, h))
                except ValueError:
                    pass
    
    if valid_images:
        # Sort by area (largest first)
        valid_images.sort(reverse=True)
        best_image = valid_images[0]
        area, src, w, h = best_image
        full_url = urljoin("https://wikimon.net", src)
        print(f"    ✅ Best image found: {full_url} ({w}x{h}, area: {area})")
        return full_url
    
    print("    ❌ No valid images found")
    return None

def is_placeholder(url):
    """Check if the URL is likely a placeholder image."""
    placeholder_indicators = [
        'placeholder', 'no-image', 'missing', 'error', 'blank',
        'default', 'none', 'null', 'undefined'
    ]
    url_lower = url.lower()
    return any(indicator in url_lower for indicator in placeholder_indicators)

def is_valid_image_url(url):
    """Check if the URL looks like a valid image."""
    # Skip data URLs, very small images, and obvious placeholders
    if url.startswith('data:'):
        return False
    
    # Check file extension
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    
    if not any(path.endswith(ext) for ext in valid_extensions):
        return False
    
    return True

def download_image_from_page_debug(digimon_page_url):
    """Debug version of image download with detailed logging."""
    print(f"Processing: {digimon_page_url}")
    try:
        response = requests.get(digimon_page_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Find the best image on the page
        image_url = find_best_image_debug(soup, digimon_page_url)
        
        if not image_url:
            print(f"  -> Could not find a valid image on this page.")
            return False

        # Get the Digimon's name for the filename
        digimon_name = digimon_page_url.split('/')[-1]
        
        # Clean the filename
        import re
        digimon_name = re.sub(r'[<>:"/\\|?*]', '_', digimon_name)
        
        # Determine file extension
        parsed_url = urlparse(image_url)
        path = parsed_url.path.lower()
        if path.endswith('.png'):
            extension = '.png'
        elif path.endswith('.jpg') or path.endswith('.jpeg'):
            extension = '.jpg'
        elif path.endswith('.gif'):
            extension = '.gif'
        else:
            extension = '.png'  # default
        
        file_name = f"{digimon_name}{extension}"
        file_path = os.path.join("digimon_images", file_name)
        
        # Download the image
        print(f"  -> Downloading image for {digimon_name}...")
        print(f"  -> Image URL: {image_url}")
        
        image_response = requests.get(image_url, stream=True, headers=HEADERS, timeout=30)
        image_response.raise_for_status()
        
        # Check if the downloaded image is actually valid
        content_length = len(image_response.content)
        print(f"  -> Downloaded {content_length} bytes")
        
        if content_length < 1000:  # Skip very small files (likely placeholders)
            print(f"  -> Skipping: Image too small ({content_length} bytes) - likely placeholder")
            return False
        
        with open(file_path, 'wb') as f:
            for chunk in image_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  -> Saved as {file_name} ({content_length} bytes)")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  -> Error processing page {digimon_page_url}: {e}")
        return False
    except Exception as e:
        print(f"  -> An unexpected error occurred: {e}")
        return False

def main():
    """Test the debug scraper on one Digimon."""
    test_digimon = "Agumon"
    page_url = f"https://wikimon.net/{test_digimon}"
    
    print("=== Debug Digimon Scraper ===")
    print(f"Testing on: {test_digimon}")
    
    success = download_image_from_page_debug(page_url)
    if success:
        print(f"\n✅ Successfully downloaded {test_digimon}")
    else:
        print(f"\n❌ Failed to download {test_digimon}")

if __name__ == "__main__":
    main()
