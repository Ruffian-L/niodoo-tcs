import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Test headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}

def debug_digimon_page(digimon_name="Agumon"):
    """Debug a single Digimon page to see what images are available."""
    base_url = "https://wikimon.net"
    page_url = f"{base_url}/{digimon_name}"
    
    print(f"=== Debugging {digimon_name} page ===")
    print(f"URL: {page_url}")
    
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Look for all images on the page
        all_images = soup.find_all('img')
        print(f"\nFound {len(all_images)} total images on the page:")
        
        for i, img in enumerate(all_images[:10]):  # Show first 10
            src = img.get('src', 'No src')
            alt = img.get('alt', 'No alt')
            width = img.get('width', 'No width')
            height = img.get('height', 'No height')
            classes = img.get('class', [])
            
            print(f"\nImage {i+1}:")
            print(f"  Src: {src}")
            print(f"  Alt: {alt}")
            print(f"  Size: {width}x{height}")
            print(f"  Classes: {classes}")
            
            # Check if it's a placeholder
            if src and len(src) < 100:  # Very short URLs might be placeholders
                print(f"  ⚠️  Short URL - might be placeholder")
        
        # Look specifically for the main image
        print(f"\n=== Looking for main image ===")
        
        # Method 1: infobox image
        infobox = soup.find('table', class_='infobox')
        if infobox:
            print("Found infobox table")
            infobox_images = infobox.find_all('img')
            for img in infobox_images:
                src = img.get('src')
                if src:
                    full_url = urljoin(base_url, src)
                    print(f"  Infobox image: {full_url}")
        else:
            print("No infobox table found")
        
        # Method 2: image class
        image_links = soup.find_all('a', class_='image')
        if image_links:
            print(f"Found {len(image_links)} image links")
            for link in image_links:
                img = link.find('img')
                if img:
                    src = img.get('src')
                    if src:
                        full_url = urljoin(base_url, src)
                        print(f"  Image link: {full_url}")
        else:
            print("No image links found")
            
    except Exception as e:
        print(f"Error: {e}")

def test_image_download(image_url):
    """Test downloading a specific image URL."""
    print(f"\n=== Testing image download ===")
    print(f"URL: {image_url}")
    
    try:
        response = requests.get(image_url, headers=HEADERS, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"Content-Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                print("✅ Valid image content type")
            else:
                print(f"⚠️  Not an image: {content_type}")
            
            # Check file size
            if len(response.content) < 1000:
                print("⚠️  Very small file - likely placeholder")
            elif len(response.content) < 10000:
                print("⚠️  Small file - might be low quality")
            else:
                print("✅ Reasonable file size")
                
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Debug a specific Digimon page
    debug_digimon_page("Agumon")
    
    # Test a specific image URL if you have one
    # test_image_download("https://wikimon.net/images/example.png")
