from chatbot_base import ChatbotBase
import faiss
import json
import torch
import os
import random
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Constants
INDEX_FOLDER = "index_db"

class VisualArchiveBot(ChatbotBase):
    def __init__(self):
        # 1. Strict Inheritance: Call the parent constructor
        super().__init__(name="Visual Archive Explorer")
        
        # 2. Load the Visual Brain (CLIP + FAISS)
        print("Loading Visual Search Engine...")
        self.is_ready = False
        try:
            # FIX: Use safetensors=False to match the builder script
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Load Index if it exists
            if os.path.exists(f"{INDEX_FOLDER}/image_vectors.index"):
                self.index = faiss.read_index(f"{INDEX_FOLDER}/image_vectors.index")
                with open(f"{INDEX_FOLDER}/metadata.json", 'r') as f:
                    self.metadata = json.load(f)
                self.is_ready = True
            else:
                print("Index missing. Run archive_builder.py first.")
                
        except Exception as e:
            print(f"Error initializing bot: {e}")

    def process_input(self, user_input):
        """
        Overrides ChatbotBase.
        Input: Dictionary {'image': PIL_Image}
        Output: Dictionary with 'matches' and 'analysis_text'
        """
        if not self.is_ready:
            return {"error": "My database is offline. Please run the builder script."}

        query_image = user_input.get('image')
        if not query_image:
            return {"error": "I need an image to see!"}

        try:
            # 1. See the Image (CLIP Vectorization)
            inputs = self.processor(images=query_image, return_tensors="pt", padding=True)
            with torch.no_grad():
                # Get features safely
                query_vector = self.model.get_image_features(**inputs)
                
                # --- THE SAFETY FIX ---
                # If it returns a "Box" instead of numbers, open the box.
                if hasattr(query_vector, 'image_embeds'):
                    query_vector = query_vector.image_embeds
                elif hasattr(query_vector, 'pooler_output'):
                    query_vector = query_vector.pooler_output
                
                # Now we definitely have a Tensor
                query_vector = query_vector / query_vector.norm(p=2, dim=-1, keepdim=True)
                query_np = query_vector.numpy()

            # 2. Search the Archive (FAISS)
            D, I = self.index.search(query_np, 5) # Get top 5 matches
            
            matches = []
            detected_styles = []

            for i, idx in enumerate(I[0]):
                if idx == -1: continue
                data = self.metadata[idx]
                score = round(D[0][i] * 100, 2)
                
                matches.append({
                    "artist": data.get('artist', 'Unknown'),
                    "tag": data.get('tag', 'General'),
                    "path": data['path'],
                    "score": score
                })
                detected_styles.append(data.get('tag'))

            # 3. Formulate an "Opinion"
            dominant_style = max(set(detected_styles), key=detected_styles.count) if detected_styles else "Unknown"
            
            return {
                "matches": matches, 
                "dominant_style": dominant_style
            }

        except Exception as e:
            return {"error": f"I had trouble seeing that: {str(e)}"}

    def generate_response(self, processed_input):
        """
        Overrides ChatbotBase.
        Generates a conversational response based on the visual findings.
        """
        if "error" in processed_input:
            return processed_input["error"]
            
        matches = processed_input.get('matches', [])
        style = processed_input.get('dominant_style', 'art')
        
        if not matches:
            return "I couldn't find anything similar in the archive. It's truly unique!"
        
        # Conversational Logic
        top_match = matches[0]
        phrases = [
            f"This piece reminds me of the **{style}** style.",
            f"I see strong visual parallels with **{style}** works.",
            f"The textures here are very similar to our **{style}** collection."
        ]
        
        intro = random.choice(phrases)
        details = f"The closest match is a piece by **{top_match['artist']}** ({top_match['score']}% similarity)."
        
        return f"{intro} {details} Here are the visual comparisons:"
