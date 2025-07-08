# src/rag_pipeline.py

"""
📦 Task 3: Building the RAG Core Logic and Evaluation

This module builds the Retrieval-Augmented Generation (RAG) pipeline:
1. Load FAISS index and metadata
2. Embed a user query
3. Retrieve top-k relevant complaint chunks
4. Construct a prompt
5. Generate an answer using an LLM

Default LLM: google/flan-t5-base (lightweight for CPU use)
Optional: mistralai/Mistral-7B-Instruct-v0.1 (for Colab/GPU — see comments)
"""

# ✅ Imports
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from scipy.spatial.distance import cosine

# ✅ Absolute Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_dir = os.path.join(project_root, 'vector_store')
index_path = os.path.join(vector_dir, 'faiss_index')
metadata_path = os.path.join(vector_dir, 'metadata.pkl')

# ✅ Config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5  # Number of relevant chunks to retrieve
TOP_K_FINAL = 3  # ✅ Use only top 3 after re-ranking

# ✅ Step 1: Load Vector Index and Metadata
def load_vector_store():
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ✅ Step 2: Initialize Embedding and Generator Models
def initialize_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ✅ Lightweight LLM for CPU (no login needed)
    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    # ❗ Optional (for Google Colab with GPU access):
    # To use Mistral 7B model instead:
    # 1. Upload your Hugging Face token to Colab:
    #    from huggingface_hub import login
    #    login(token="your_hf_token")
    # 2. Install: !pip install transformers accelerate bitsandbytes
    # 3. Replace this line:
    #    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")

    return embedder, generator

# ✅ Step 3: Clean chunk text
# ✅ Implemented for Task 3 Improvement ✅
def clean_chunk(text, max_words=300):
    text = text.replace("\n", " ").replace("xxxx", "").strip()
    words = text.split()
    return " ".join(words[:max_words])

# ✅ Step 4: Retrieve, Re-rank, and Clean Top Context Chunks
# ✅ Improved for Task 3 quality

def retrieve_context(query, embedder, index, metadata, top_k=TOP_K, final_k=TOP_K_FINAL):
    query_vector = embedder.encode([query])[0]
    distances, indices = index.search(np.array([query_vector]), top_k)

    scored_chunks = []
    for i in range(top_k):
        idx = indices[0][i]
        meta = metadata[idx]
        chunk_text = clean_chunk(meta["text"])  # ✅ Clean the chunk text
        chunk_vector = embedder.encode([chunk_text])[0]
        similarity = 1 - cosine(query_vector, chunk_vector)

        scored_chunks.append((similarity, chunk_text, meta))

    # ✅ Sort by similarity descending and keep top final_k
    top_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)[:final_k]

    context_texts = [chunk[1] for chunk in top_chunks]
    meta_used = [chunk[2] for chunk in top_chunks]

    return "\n\n".join(context_texts), meta_used

# ✅ Step 5: Build Prompt for the LLM
def build_prompt(context, query):
    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say: 'I don't have enough information.'

Context:
{context}

Question: {query}
"""
    return prompt

# ✅ Step 6: Run the RAG Pipeline
def answer_query(query, embedder, generator, index, metadata, top_k=TOP_K):
    context, retrieved_meta = retrieve_context(query, embedder, index, metadata, top_k)
    prompt = build_prompt(context, query)

    # flan-t5 uses "text2text-generation" and expects prompt only
    response = generator(prompt, max_new_tokens=256)[0]["generated_text"].strip()

    return response, context, retrieved_meta

# ✅ Example Usage for Evaluation
if __name__ == "__main__":
    print("\n🚀 Running the RAG pipeline for evaluation...\n")

    # Load vector data & models
    index, metadata = load_vector_store()
    embedder, generator = initialize_models()

    # Sample test questions (can be extended to 5–10 in evaluation)
    questions = [
        "What are common issues reported about BNPL?",
        "Why are customers frustrated with credit card services?",
        "Do people complain about transfer delays?",
        "Are there complaints related to loan application processes?",
        "What problems do users face with mobile banking apps?"
    ]

    for q in questions:
        print(f"🔹 Question: {q}")
        answer, context, meta = answer_query(q, embedder, generator, index, metadata)
        print(f"🧠 Answer: {answer}\n")
        print(f"📄 Retrieved Source Sample: {meta[0]['text'][:200]}...\n")
        print("-" * 80)
