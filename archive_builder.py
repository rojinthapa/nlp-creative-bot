import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
IMAGE_FOLDER = "images"
INDEX_FOLDER = "index_db"
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Define Zero-Shot Labels for Auto-Tagging
STYLE_TAGS = ["Oil Painting", "Sketch", "Photography", "Sculpture", "Digital Art", "Abstract"]

def build_archive():
    print("--- 1. Loading AI Model (CLIP) ---")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"--- 2. Scanning '{IMAGE_FOLDER}' ---")
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"Created '{IMAGE_FOLDER}'. Please add images and run this script again.")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found! Add .jpg or .png files to the 'images' folder.")
        return

    # Initialize FAISS Index (Vector DB)
    d = 512  # CLIP output dimension
    index = faiss.IndexFlatIP(d) 
    metadata = []
    
    print(f"--- 3. Processing {len(image_files)} images ---")
    for i, filename in enumerate(image_files):
        path = os.path.join(IMAGE_FOLDER, filename)
        try:
            image = Image.open(path)
            
            # A. Generate Vector
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            # Normalize for Cosine Similarity
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            # B. Add to Index
            index.add(features.numpy())
            
            # C. Auto-Tagging (Zero-Shot Classification)
            text_inputs = processor(text=STYLE_TAGS, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**text_inputs)
            best_tag_idx = outputs.logits_per_image.argmax().item()
            detected_style = STYLE_TAGS[best_tag_idx]
            
            # D. Save Metadata
            meta = {
                "id": i,
                "filename": filename,
                "path": path,
                "tag": detected_style,
                "artist": "Unknown Artist", # Placeholder
                "year": "N/A"
            }
            metadata.append(meta)
            print(f"[{i+1}/{len(image_files)}] Indexed: {filename} -> Tag: {detected_style}")
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # Save to disk
    faiss.write_index(index, f"{INDEX_FOLDER}/image_vectors.index")
    with open(f"{INDEX_FOLDER}/metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    print("--- Done! Database built successfully. ---")

if __name__ == "__main__":
    build_archive()