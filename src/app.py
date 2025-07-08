# --- 1. Imports ---
# Core libraries for file paths, data handling, and the AI models
import os
import faiss
import pickle
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- 2. Configuration and Paths ---
# This section centralizes all the important variables and file paths,
# making the code clean and easy to modify.

# --- ✅ Absolute Paths ---
# We define the absolute path to the project root to ensure that the script
# can find the vector_store directory no matter where you run it from.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_dir = os.path.join(project_root, 'vector_store')
index_path = os.path.join(vector_dir, 'faiss_index')
metadata_path = os.path.join(vector_dir, 'metadata.pkl')

# --- ✅ Model and RAG Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-base"
TOP_K = 5 # The number of initial documents to retrieve from FAISS.

# --- 3. Loading Models and Vector Store ---
# These functions are designed to load everything into memory once when the app starts.
# This is efficient because we don't have to reload the models for every user query.

print("Initializing models and loading vector store...")

# --- ✅ Step 1: Initialize Embedding and Generator Models ---
# This function loads the Sentence Transformer for embeddings and the T5 model for generation.
def initialize_models():
    """Loads the embedding and text generation models from Hugging Face."""
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Using a text2text-generation pipeline is ideal for question-answering models like Flan-T5.
    generator = pipeline("text2text-generation", model=GENERATOR_MODEL_NAME)
    return embedder, generator

# --- ✅ Step 2: Load FAISS Vector Index and Metadata ---
# This function loads the pre-computed FAISS index and the corresponding metadata (the actual text chunks).
def load_vector_store():
    """Loads the FAISS index and metadata from disk."""
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# --- Load all components into global variables for the app to use ---
embedder, generator = initialize_models()
faiss_index, metadata = load_vector_store()

print("Models and vector store loaded successfully!")


# --- 4. The Core RAG Logic ---
# This is the main function that processes a user's query.

def get_rag_response(question, chat_history):
    """
    This function takes a user's question and chat history, performs the RAG process,
    and returns the generated answer and the sources used.
    """
    # --- Step 4a: Embed the User's Question ---
    # We convert the user's question into a vector so we can search for it in FAISS.
    question_embedding = embedder.encode([question])

    # --- Step 4b: Search FAISS for Relevant Chunks ---
    # We use the FAISS index to find the top_k most similar text chunks.
    # The search returns the distances and the indices of the matching vectors.
    distances, indices = faiss_index.search(np.array(question_embedding).astype('float32'), TOP_K)

    # --- Step 4c: Retrieve the Actual Text Chunks ---
    # We use the retrieved indices to look up the original text chunks from our metadata.

    retrieved_chunks = [metadata[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    # --- Step 4d: Engineer the Prompt for the LLM ---
    # This is a crucial step. We create a detailed prompt that tells the LLM
    # exactly how to behave. It's instructed to act as a financial analyst
    # and ONLY use the provided context.
    prompt_template = f"""
    You are a financial analyst assistant for CrediTrust.
    Your task is to answer the user's question based ONLY on the following context of customer complaints.
    If the context does not contain the answer, state that you don't have enough information. Do not use any external knowledge.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    # --- Step 4e: Generate the Answer ---
    # We pass the engineered prompt to our generator model.
    generated_output = generator(prompt_template, max_length=512)
    answer = generated_output[0]['generated_text']

    # --- Step 4f: Format the Sources for Display ---
    # We create a nicely formatted string with the source chunks to show the user.
    # This is critical for building trust and allowing verification.
    source_info = "\n--- SOURCES ---\n"
    for i, chunk in enumerate(retrieved_chunks):
        source_info += f"Source [{i+1}]:\n> {chunk}\n\n"

    # --- Combine the answer and sources ---
    full_response = f"{answer}\n\n{source_info}"

    return full_response


# --- 5. Building the Gradio User Interface ---
# This is where we define the layout and functionality of our web app.

with gr.Blocks(theme=gr.themes.Soft(), title="CrediTrust Complaint Analyst") as app:
    gr.Markdown("# CrediTrust AI Complaint Analyst 🤖")
    gr.Markdown("Ask questions about customer complaints and get synthesized, evidence-backed answers.")

    # The gr.ChatInterface is a high-level component that creates a full chat UI.
    # We pass our main RAG function `get_rag_response` to it.
    chatbot = gr.ChatInterface(
        fn=get_rag_response,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="e.g., Why are people unhappy with BNPL?", container=False, scale=7),
        examples=[
            "Why are people unhappy with BNPL?",
            "What are the common issues with money transfers?",
            "Are there complaints about unexpected credit card fees?"
        ],
        cache_examples=False
    )

    # This creates a "Clear" button.
    # The .click() method defines what happens when the button is clicked.
    # Here, it does nothing (`fn=None`) but clears the chatbot component (`outputs=[chatbot]`).
    # The `js` argument is a bit of JavaScript to make it work smoothly.
    clear_button = gr.Button("🗑️ Clear Chat")
    clear_button.click(fn=None, inputs=None, outputs=[chatbot], js="() => { return [] }")


# --- 6. Launching the App ---
# The .launch() method starts the web server and makes the UI accessible in your browser.
if __name__ == "__main__":
    app.launch(debug=True) # Use debug=True for development to see errors.

