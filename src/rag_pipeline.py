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