# src/chunk_embed_index.py

"""
📦 Task 2: Text Chunking, Embedding, and Vector Store Indexing

This script performs the following steps:
1. Load cleaned complaint narratives.
2. Split the text into manageable chunks with overlap.
3. Generate vector embeddings for each chunk.
4. Store them in a searchable vector database using FAISS.

Directory structure:
- Input: /data/filtered_complaints.csv
- Output: /vector_store/faiss_index and /vector_store/metadata.pkl
"""

# ✅ Imports
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# ✅ Project Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'data', 'processed', 'filtered_complaints.csv')
vector_dir = os.path.join(project_root, 'vector_store')
vector_index_path = os.path.join(vector_dir, 'faiss_index')
metadata_path = os.path.join(vector_dir, 'metadata.pkl')

# ✅ Step 1: Load Cleaned Data
def load_cleaned_data(filepath):
    return pd.read_csv(filepath)

# ✅ Step 2: Initialize Text Splitter
def create_text_splitter(chunk_size, overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )

# ✅ Step 3: Initialize Embedding Model
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

# ✅ Step 4: Generate Embeddings and Metadata
def generate_embeddings(df, splitter, embedder):
    all_embeddings = []
    metadata_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        complaint_id = row.get("Complaint ID", idx)  # fallback if missing
        product = row["Product"]
        text = row["cleaned_narrative"]

        chunks = splitter.split_text(text)
        embeddings = embedder.encode(chunks)

        for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            all_embeddings.append(vector)
            metadata_records.append({
                "complaint_id": complaint_id,
                "product": product,
                "chunk_index": i,
                "text": chunk_text
            })

    return all_embeddings, metadata_records