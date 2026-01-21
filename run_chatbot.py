import streamlit as st
from visual_archive_bot import VisualArchiveBot
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="UCA Visual Search", page_icon="🏛️", layout="centered")

# --- CSS STYLING (To make it look pro) ---
st.markdown("""
<style>
    .stChatMessage {background-color: #f0f2f6; border-radius: 10px; padding: 10px;}
    .stImage {border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if "bot" not in st.session_state:
    with st.spinner("Waking up the Curator..."):
        st.session_state.bot = VisualArchiveBot()
        
    # Start the conversation history
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Digital Archive. Upload an image, and I will analyze its style to find historical connections."}
    ]

bot = st.session_state.bot

# --- HEADER ---
st.title("🏛️ AI Art Curator")
st.caption("Powered by CLIP + FAISS | Visual Semantic Search")

# --- CHAT LOOP ---
# 1. Display existing history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If the message has images attached (previous results), show them
        if "images" in msg:
            cols = st.columns(len(msg["images"]))
            for i, img_data in enumerate(msg["images"]):
                with cols[i]:
                    st.image(img_data["path"], caption=f"{img_data['score']}% Match")

# 2. User Input Area
uploaded_file = st.file_uploader("Upload an artwork or photo...", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if uploaded_file:
    # A. Display User's Image in Chat
    user_image = Image.open(uploaded_file)
    with st.chat_message("user"):
        st.image(user_image, caption="I'm looking for this style.", width=250)
    
    # Add to history so it stays on screen
    st.session_state.messages.append({"role": "user", "content": "I'm looking for this style."})

    # B. Bot Processing
    if bot.is_ready:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing textures, color, and composition..."):
                
                # CALL THE CLASS LOGIC
                raw_input = {'image': user_image}
                results = bot.process_input(raw_input)
                
                # GENERATE TEXT RESPONSE
                response_text = bot.generate_response(results)
                st.write(response_text)
                
                # VISUAL RESULTS
                if "matches" in results and results["matches"]:
                    matches = results["matches"][:3] # Show top 3
                    cols = st.columns(len(matches))
                    for i, match in enumerate(matches):
                        with cols[i]:
                            st.image(match['path'], use_container_width=True)
                            st.caption(f"{match['tag']}\n{match['score']}%")
                    
                    # Save context to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "images": matches
                    })
    else:
        st.error("Error: Database not found. Please run 'archive_builder.py' first.")