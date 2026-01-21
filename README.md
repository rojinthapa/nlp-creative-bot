# Visual Archive Retriever (AI Art Curator)

A visual similarity search engine that allows users to retrieve artworks from a digital archive using semantic image analysis.

## Features
* **Visual Query:** Upload an image to find stylistically similar works.
* **AI-Powered:** Uses OpenAI's **CLIP** model for vector embeddings.
* **High Performance:** Uses **FAISS** for millisecond-speed retrieval.
* **Interactive Chat:** A conversational agent that acts as an Art Curator.

## Architecture
1.  **Stage 1 (Offline):** `archive_builder.py` scans images, generates vectors, and builds a FAISS index.
2.  **Stage 2 (Online):** `run_chatbot.py` takes user input, vectorizes it, and queries the index.

## Installation
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Download the dataset (500 images):
    ```bash
    python download_images.py
    ```
3.  Build the database:
    ```bash
    python archive_builder.py
    ```

## Usage
Run the visual interface:
```bash
streamlit run run_chatbot.py