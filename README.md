# StepStride: Machine Learning Shoe Search with CLIP

StepStride is a semantic search engine specifically designed for shoe catalogs. Unlike traditional keyword searches, this project uses **Machine Learning** via the **CLIP (Contrastive Language-Image Pre-training)** model to understand the visual and textual relationship between descriptions and products.

## 🚀 Project Overview
This project implements a high-performance similarity search. It converts images of shoes and text queries into numerical vectors (embeddings). By calculating the cosine similarity between these vectors, the system can find the most relevant shoes even for complex or abstract text queries like "running shoes for professional athletes" or "vintage leather boots."

## 📊 Dataset
The project utilizes the **Fashion Images** dataset available on Kaggle:
*   **Source:** [Fashion Images Dataset by Vikash Raj Luhaniwal](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images)
*   **Focus:** For this implementation, the system is optimized and filtered to process and rank shoe-related imagery.

## 🛠️ How it Works
1.  **Image Embedding:** Every image in the dataset is processed through a pre-trained `CLIPModel` to extract its visual features.
2.  **Batch Processing:** To handle large datasets efficiently, images are processed in batches and stored as `.pt` (PyTorch) files.
3.  **Semantic Search:** When a user enters a text query, the system generates a text embedding and compares it against all image embeddings using matrix multiplication (Cosine Similarity).
4.  **Ranking & Display:** The system ranks the images from highest to lowest similarity and displays a grid of the top matches.

## 💻 Core Technologies
*   **PyTorch:** For tensor operations and model handling.
*   **Hugging Face Transformers:** To access the `openai/clip-vit-base-patch32` model.
*   **NumPy:** For efficient indexing and sorting of scores.
*   **Torchvision:** For image transformations and grid visualization.
