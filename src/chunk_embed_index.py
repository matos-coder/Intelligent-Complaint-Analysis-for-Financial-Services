# src/chunk_embed_index.py

"""
📦 Task 2: Text Chunking, Embedding, and Vector Store Indexing

This script performs the following steps:
1. Load cleaned complaint narratives.
2. Split the text into manageable chunks with overlap.
3. Generate vector embeddings for each chunk.
4. Store them in a searchable vector database using FAISS.

Directory structure:
- Input: data/filtered_complaints.csv
- Output: vector_store/faiss_index
"""

# ✅ Imports
import pandas as pd
import os
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# ✅ Constants
INPUT_DATA_PATH = "data/filtered_complaints.csv"
VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150