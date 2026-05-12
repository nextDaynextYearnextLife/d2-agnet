import requests
import os
from PIL import Image
from io import BytesIO

# Create directory for hero images
os.makedirs('visualization/hero_images', exist_ok=True)

# Fetch hero data from OpenDota API
hero_url = "https://api.opendota.com/api/heroes"
response = requests.get(hero_url)
hero_data = response.json()

# Download hero images
for hero in hero_data:
    hero_id = hero['id']
    hero_name = hero['localized_name']
    # Build image URL from hero name (OpenDota format)
    name = hero['name'].replace('npc_dota_hero_', '')
    image_url = f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{name}_lg.png"
    
    try:
        img_response = requests.get(image_url, timeout=10)
        if img_response.status_code == 200:
            img = Image.open(BytesIO(img_response.content))
            
            # Save image
            img_path = f'visualization/hero_images/{hero_id}_{name}.png'
            img.save(img_path)
            print(f"Downloaded: {img_path}")
        else:
            print(f"Failed to download {hero_name}: HTTP {img_response.status_code}")
    except Exception as e:
        print(f"Failed to download {hero_name}: {e}")

print("Hero images download completed.")
