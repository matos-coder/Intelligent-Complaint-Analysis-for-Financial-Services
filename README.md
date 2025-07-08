# Project Name

intelligent-complaint-analysis

## Project Overview

This repository contains a semantic AI-powered complaint analysis pipeline for CrediTrust Financial, focused on turning thousands of unstructured customer complaints into actionable insights. The system aims to support internal teams—like product managers, support analysts, and compliance officers—by enabling natural language querying of real customer pain points across financial services. The project includes:

- A robust and modular data preprocessing pipeline.
- Exploratory Data Analysis (EDA) to inspect the quality and structure of customer complaints.
- A vector-based semantic retrieval pipeline powered by FAISS and Sentence Transformers.
- A complete Retrieval-Augmented Generation (RAG) pipeline that uses a Large Language Model (LLM) to generate insightful answers.
- An interactive web-based chatbot interface built with Gradio for easy, non-technical user access.

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
Provide a simple web-based interface for non-technical users to query the RAG system.

**Location**:
Implemented in `app.py`.

**Features**:

- Uses `Gradio` for front-end UI.
- Preloads sample questions.
- Displays source chunks for transparency.
- Includes "Clear Chat" button for resetting session.

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
