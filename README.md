# Project Name

intelligent-complaint-analysis

## Project Overview

This repository contains a complete end-to-end pipeline for Intelligent Complaint Analysis designed for **CrediTrust Financial**. It leverages the power of Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) to convert thousands of unstructured customer complaints into **actionable insights**.

The goal is to enable internal stakeholders (product managers, analysts, compliance officers) to:
- Query customer complaints using natural language
- Surface hidden patterns in user pain points
- Support evidence-based decision-making and customer satisfaction improvements

The project includes:

- ✅ A robust and modular **data preprocessing** pipeline
- ✅ Thorough **Exploratory Data Analysis (EDA)** to inspect the structure and patterns in complaints
- ✅ A **semantic vector-based retrieval system** using FAISS and Sentence Transformers
- ✅ A **Retrieval-Augmented Generation (RAG)** pipeline to retrieve, ground, and generate answers
- ✅ An **interactive Gradio chatbot interface** that makes querying seamless for non-technical users

This project serves as a practical foundation for building **AI assistants in financial services** to reduce response times, improve compliance monitoring, and enhance user research.


## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/matos-coder/Intelligent-Complaint-Analysis-for-Financial-Services
cd Intelligent-Complaint-Analysis-for-Financial-Services
```

### Create a Virtual Environment

```bash
python -m venv .venv
venv\Scripts\activate          # For Windows
# OR
source venv/bin/activate       # For Linux/macOS
```

### Install Required Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
intelligent-complaint-analysis/
├── data/
│   └── filtered_complaints.csv        # Preprocessed dataset
├── notebooks/
│   └── EDA.ipynb                    # Initial data analysis and insights
├── src/
│   ├── chunk_embed_index.py         # Task 2: Chunking, embedding, and vector store creation
│   ├── rag_evaluation.py            # Task 3: RAG pipeline logic and qualitative evaluation
│   └── app.py                        # Task 4: Gradio chatbot application
├── vector_store/
│   ├── faiss_index                  # FAISS vector index
│   └── metadata.pkl                 # Metadata (chunk -> source link)
├── requirements.txt
└── README.md
```

## Tasks

### Task 1: Exploratory Data Analysis (EDA) and Preprocessing

**Objective**:
Understand the structure, content, and quality of the complaint data, and clean it for downstream semantic search and LLM processing.

**Location**:
All analysis is implemented in `notebooks/EDA.ipynb`.

**Key Steps Performed**:

- **Data Inspection**:
  - Dataset loaded from the Consumer Financial Protection Bureau (CFPB).
  - Inspected data types, null values, and initial statistics.
  - Counted number of complaints across financial products.

- **Narrative Length Analysis**:
  - Calculated word counts for each Consumer complaint narrative.
  - Visualized distribution of complaint lengths to identify very short/long cases.

- **Filtering**:
  - Focused on 5 relevant financial products:
    - Credit card
    - Personal loan
    - Buy Now, Pay Later (BNPL)
    - Savings account
    - Money transfers
  - Removed complaints with missing narratives.

- **Text Cleaning**:
  - Lowercased all narratives.
  - Removed special characters and boilerplate legal language.
  - Stripped whitespace for consistency.

- **Saving Cleaned Dataset**:
  - Final cleaned and filtered dataset saved as `data/filtered_complaints.csv`.

**Key Insights from EDA**:
- Most complaints were about BNPL and Credit Cards, indicating high user dissatisfaction.
- Missing narratives were excluded from training.
- Long-tail distribution of narrative lengths revealed highly detailed complaints.

---

### Task 2: Text Chunking, Embedding, and Vector Indexing

**Objective**:
Convert the cleaned complaints into dense vector representations suitable for efficient semantic search using Retrieval-Augmented Generation (RAG).

**Location**:
Implemented in `src/chunk_embed_index.py`.

**Components and Workflow**:

- **Chunking**:
  - Used LangChain’s `RecursiveCharacterTextSplitter`.
  - `chunk_size = 800` and `chunk_overlap = 150` for semantic cohesion.

- **Embedding**:
  - Used `sentence-transformers/all-MiniLM-L6-v2`.
  - Lightweight, efficient, accurate model suitable for CPU.

- **Metadata Mapping**:
  - Stored chunk text and product metadata to support explainability.

- **Vector Store Indexing**:
  - Built FAISS index.
  - Saved to `vector_store/faiss_index` and `vector_store/metadata.pkl`.

**Justifications**:
- MiniLM was chosen for local compatibility and balanced performance.
- Chunking ensured longer narratives were preserved without loss of meaning.

---

### Task 3: Building the RAG Core Logic and Evaluation

**Objective**:
Construct and evaluate a Retrieval-Augmented Generation pipeline combining semantic retrieval, prompt engineering, and text generation.

**Location**:
Implemented in `src/rag_evaluation.py`.

**Key Components**:

- **Retriever**: Uses FAISS + MiniLM embeddings.
- **Prompt**: Instructions to the LLM to act as a CrediTrust analyst and use only given context.
- **Generator**: Uses `google/flan-t5-base` for generation.
- **Post-processing**: Cosine similarity re-ranking + chunk cleaning.

**Evaluation Table**:

| Question                                                | Generated Answer                                                                                                                                    | Retrieved Sources (Sample)                                                                                                     | Quality Score (1-5) | Comments/Analysis                                                |
|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|---------------------|------------------------------------------------------------------|
| Why are people unhappy with BNPL?                       | BNPL has been closing customer accounts without providing clear reasons which has affected customers who have been using their cash back credit cards... | ...political motivations particularly in light of tensions between xxxx and the united states...                              | 4                   | Answer directly uses sources, coherent, and well-grounded        |
| What are the common issues with money transfers?        | A western union money transfer was blocked... dollar amount of xxxx has not been credited... immigrants and public policy currently opposes immigrants... | ...nativeborn united states of america citizen who was trying to use this service to send slightly more than a xxxx dollars... | 4                   | Relevant, covers policy and technical issues, slightly long      |
| Are there complaints about unexpected credit card fees? | Yes. Charges were issued along with significant late fees... companies may exploit elderly customers...                                              | ...credit card company charged me a fee of on xxxx without any indication what that fee is for...                             | 5                   | Very strong match between complaint context and generated response |

---

### Task 4: Creating an Interactive Chat Interface

**Objective**:
Deploy a simple and intuitive web interface that allows business users to ask questions in natural language and receive complaint-based answers generated by the RAG pipeline.

**Location**:
Implemented in `src/app.py` using the **Gradio** framework.

**Key Features**:

- 💬 **Natural Language Query Input**: Users can enter questions like "Why are people unhappy with BNPL?".
- 🔍 **LLM-Powered Answers**: Behind the scenes, the app retrieves the top relevant complaint chunks and uses a language model to generate grounded responses.
- 📄 **Source Transparency**: The interface displays the top retrieved complaint texts that were used to formulate the answer.
- 🔄 **Clear Session Option**: Allows users to reset the chat state.
- 🧠 **Plug-and-Play RAG Stack**: Loads models and vector index during startup for fast querying.

**How It Works**:
1. Loads the FAISS vector index and metadata.
2. Accepts a user query and computes the embedding.
3. Performs semantic retrieval and re-ranking of complaint chunks.
4. Constructs a structured prompt for the LLM.
5. Uses the model (`flan-t5-base`) to generate a response.
6. Displays both the answer and the complaint sources.

This chatbot enables:
- **Fast insight retrieval** for research and analysis
- **Transparent evidence** from original customer narratives
- **Accessibility** to stakeholders without technical backgrounds

The app is lightweight and deployable locally or as a hosted microservice.

---

## How to Run the Pipeline

**Task 1 - Data Preprocessing**:

```bash
Run inside Jupyter Notebook: notebooks/EDA.ipynb
```

**Task 2 - Chunking & Indexing**:

```bash
python src/chunk_embed_index.py
```

**Task 3 - RAG Evaluation**:

```bash
python src/rag_evaluation.py
```

**Task 4 - Launch Chatbot Interface**:

```bash
python src/app.py
```

Then navigate to [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Requirements

```bash
pip install -r requirements.txt
```

**Includes**:

- pandas
- numpy
- tqdm
- langchain
- sentence-transformers
- faiss-cpu
- transformers
- gradio

## Contact

For questions, feedback, or contributions, feel free to open an issue or submit a pull request.
