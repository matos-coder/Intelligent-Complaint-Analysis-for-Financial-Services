# src/rag_pipeline.py

"""
📦 Task 3: Building the RAG Core Logic and Evaluation

This module builds the Retrieval-Augmented Generation (RAG) pipeline:
1. Load FAISS index and metadata
2. Embed a user query
3. Retrieve top-k relevant complaint chunks
4. Construct a prompt
5. Generate an answer using an LLM

The output is a human-readable answer grounded in retrieved context.
"""

# ✅ Imports
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ✅ Absolute Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_dir = os.path.join(project_root, 'vector_store')
index_path = os.path.join(vector_dir, 'faiss_index')
metadata_path = os.path.join(vector_dir, 'metadata.pkl')

# ✅ Config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5  # Number of relevant chunks to retrieve

# ✅ Step 1: Load Vector Index and Metadata
def load_vector_store():
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ✅ Step 2: Initialize Embedding and Generator Models
def initialize_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
    return embedder, generator