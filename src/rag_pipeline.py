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

# ✅ Step 3: Retrieve Top-k Context Chunks
def retrieve_context(query, embedder, index, metadata, top_k=TOP_K):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)

    retrieved_chunks = []
    for i in range(top_k):
        idx = indices[0][i]
        meta = metadata[idx]
        chunk_text = meta["text"]
        retrieved_chunks.append(chunk_text)

    return "\n\n".join(retrieved_chunks), [metadata[i] for i in indices[0]]

# ✅ Step 4: Build Prompt for the LLM
def build_prompt(context, query):
    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say: 'I don't have enough information.'

Context:
{context}

Question: {query}
Answer:
"""
    return prompt

# ✅ Step 5: Run the RAG Pipeline
def answer_query(query, embedder, generator, index, metadata, top_k=TOP_K):
    context, retrieved_meta = retrieve_context(query, embedder, index, metadata, top_k)
    prompt = build_prompt(context, query)

    # Use a small max_new_tokens value for concise answers
    response = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"]

    # Extract only the model's answer portion
    answer = response.split("Answer:")[-1].strip()
    return answer, context, retrieved_meta

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
        "Do people complain about transfer delays?"
    ]

    for q in questions:
        print(f"🔹 Question: {q}")
        answer, context, meta = answer_query(q, embedder, generator, index, metadata)
        print(f"🧠 Answer: {answer}\n")
        print(f"📄 Retrieved Source Sample: {meta[0]['text'][:200]}...\n")
        print("-" * 80)
