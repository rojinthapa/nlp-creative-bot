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
        # FIX 1: Add use_safetensors=False to stop the network timeout crash
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"--- 2. Scanning '{IMAGE_FOLDER}' ---")
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"Created '{IMAGE_FOLDER}'.")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found!")
        return

    # Initialize FAISS Index
    d = 512
    index = faiss.IndexFlatIP(d) 
    metadata = []
    valid_vectors = []
    
    print(f"--- 3. Processing {len(image_files)} images ---")
    
    for i, filename in enumerate(image_files):
        path = os.path.join(IMAGE_FOLDER, filename)
        try:
            image = Image.open(path)
            
            # A. Generate Vector
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                
                # --- FIX 2: THE SAFETY NET ---
                # If the model returns a "Box" (BaseModelOutput), extract the numbers manually
                if hasattr(features, 'image_embeds'):
                    features = features.image_embeds
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                
            # Now 'features' is guaranteed to be a Tensor. We can do math.
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            # Store
            valid_vectors.append(features.numpy())
            
            # B. Auto-Tagging
            text_inputs = processor(text=STYLE_TAGS, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**text_inputs)
            best_tag_idx = outputs.logits_per_image.argmax().item()
            detected_style = STYLE_TAGS[best_tag_idx]
            
            # C. Save Metadata
            meta = {
                "id": i,
                "filename": filename,
                "path": path,
                "tag": detected_style,
                "artist": "Unknown Artist", 
                "year": "N/A"
            }
            metadata.append(meta)
            
            if i % 10 == 0:
                print(f"[{i+1}/{len(image_files)}] Indexed: {filename} -> Tag: {detected_style}")
            
        except Exception as e:
            # Print the error but don't crash
            # print(f"Skipping {filename}: {e}") # Uncomment to see errors
            pass

    # Batch Add to FAISS
    if valid_vectors:
        batch_matrix = np.vstack(valid_vectors)
        index.add(batch_matrix)
        
        faiss.write_index(index, f"{INDEX_FOLDER}/image_vectors.index")
        with open(f"{INDEX_FOLDER}/metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"--- Success! Indexed {len(metadata)} images. ---")
    else:
        print("--- Failed: No images were indexed. ---")

if __name__ == "__main__":
    build_archive()
