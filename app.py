import streamlit as st
import faiss
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- SETUP ---
st.set_page_config(page_title="Art Archive Visual Search", layout="wide")
INDEX_FOLDER = "index_db"

@st.cache_resource
def load_resources():
    print("Loading resources...")
    # Load Model (FIX 1: use_safetensors=False to prevent network crash)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load Database (FAISS + JSON)
    try:
        index = faiss.read_index(f"{INDEX_FOLDER}/image_vectors.index")
        with open(f"{INDEX_FOLDER}/metadata.json", 'r') as f:
            metadata = json.load(f)
    except Exception:
        return None, None, None, None

    return model, processor, index, metadata

model, processor, index, metadata = load_resources()

# --- UI HEADER ---
st.title("🏛️ Digital Archive Retriever")
st.markdown("Upload an image to find visually similar artworks using **CLIP + FAISS**.")

# --- SIDEBAR: TAG FILTERING ---
with st.sidebar:
    st.header("🔍 Refine Search")
    st.info("Uses Auto-Generated Tags")
    
    # Get all unique tags from our metadata
    if metadata:
        all_tags = sorted(list(set([item['tag'] for item in metadata])))
        selected_style = st.multiselect("Filter by Style:", all_tags)
    else:
        selected_style = []

# --- MAIN SEARCH LOGIC ---
uploaded_file = st.file_uploader("Upload Query Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file and index:
    col1, col2 = st.columns([1, 2])
    
    # 1. Process User Input
    user_image = Image.open(uploaded_file)
    with col1:
        st.image(user_image, caption="Your Query", width=250)
    
    # 2. Generate Embedding (Live)
    inputs = processor(images=user_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_vector = model.get_image_features(**inputs)
        
        # --- FIX 2: THE SAFETY NET (Unwrap the Box) ---
        if hasattr(query_vector, 'image_embeds'):
            query_vector = query_vector.image_embeds
        elif hasattr(query_vector, 'pooler_output'):
            query_vector = query_vector.pooler_output
            
        # Normalize for Cosine Similarity
        query_vector = query_vector / query_vector.norm(p=2, dim=-1, keepdim=True)
        query_np = query_vector.numpy()

    # 3. FAISS Search
    # Search for top 20 candidates (Batch Loading)
    D, I = index.search(query_np, 20) 
    
    # --- DISPLAY RESULTS ---
    with col2:
        st.subheader("Results")
        results_found = 0
        
        # Loop through results
        for i, idx in enumerate(I[0]):
            score = D[0][i] # This is the cosine similarity score
            data = metadata[idx]
            
            # FILTERING LOGIC
            if selected_style and data['tag'] not in selected_style:
                continue # Skip if tag doesn't match
                
            results_found += 1
            
            # RESULT CARD
            with st.expander(f"#{i+1} Match: {data['artist']} ({data['tag']})", expanded=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(data['path'], width=150)
                with c2:
                    # Similarity Percentage
                    sim_score = round(score * 100, 2)
                    st.metric("Similarity Match", f"{sim_score}%")
                    st.write(f"**Period:** {data['year']}")
                    st.write(f"**Archive ID:** {data['id']}")
                    st.caption(f"Detected Style: {data['tag']}")
        
        if results_found == 0:
            st.warning("No matches found with current filters.")

elif not index:
    st.error("Archive not found! Please run 'archive_builder.py' first.")
