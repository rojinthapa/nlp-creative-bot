# Visual Archive Explorer 🔎

**A Semantic Visual Search Engine for Digital Art Discovery**

This tool implements a **content-based image retrieval (CBIR)** system designed to help users navigate vast digital archives. Instead of relying on text keywords, the system uses **Computer Vision** to understand the stylistic and semantic content of an image, allowing for a "visual-to-visual" search experience.

### 🚀 Key Features
* **Visual Querying:** Upload an image to find stylistically similar artworks, sketches, or photographs.
* **Zero-Shot Learning:** Powered by **OpenAI's CLIP** model, enabling the system to understand image concepts without manual labeling.
* **Fast Indexing:** Utilizes **FAISS** for millisecond-latency retrieval from large datasets.
* **Interactive Agent:** Includes a conversational AI interface that acts as a digital historian, analyzing and explaining visual connections.

### 🛠️ Technical Architecture
The system follows a two-stage retrieval pipeline:
1.  **Offline Vectorization:** The `archive_builder.py` script scans the dataset and converts images into 512-dimensional vector embeddings.
2.  **Real-Time Inference:** The `run_chatbot.py` application computes Cosine Similarity between the user's input and the stored archive index.

---

### 📋 How to Run
1.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Dataset:**
    ```bash
    python download_images.py
    ```
    *(Downloads 500 sample images)*
3.  **Build Index:**
    ```bash
    python archive_builder.py
    ```
4.  **Launch Interface:**
    ```bash
    streamlit run run_chatbot.py
    ```
