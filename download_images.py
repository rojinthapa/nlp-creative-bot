import os
import requests
import time

# Configuration
COUNT = 500            # How many images you want
SAVE_FOLDER = "images" # Where to save them
IMG_SIZE = 300         # 300x300 pixels (Tiny but visible)

def download_dataset():
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    
    print(f"🚀 Starting download of {COUNT} images from Lorem Picsum...")
    
    for i in range(1, COUNT + 1):
        try:
            # Get a random image URL
            url = f"https://picsum.photos/{IMG_SIZE}/{IMG_SIZE}?random={i}"
            
            # Fetch the image
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                file_path = os.path.join(SAVE_FOLDER, f"art_sample_{i}.jpg")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                # Print progress every 10 images
                if i % 10 == 0:
                    print(f"[{i}/{COUNT}] Downloaded...")
            else:
                print(f"Skipped {i} (Status: {response.status_code})")
                
        except Exception as e:
            print(f"Error downloading {i}: {e}")
            
    print("✅ Download Complete! You now have a dataset.")

if __name__ == "__main__":
    download_dataset()