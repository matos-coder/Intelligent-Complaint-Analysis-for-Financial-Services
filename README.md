# Project Name
intelligent-complaint-analysis

# Project Overview

This repository contains a semantic AI-powered complaint analysis pipeline for CrediTrust Financial, focused on turning thousands of unstructured customer complaints into actionable insights. The system aims to support internal teams—like product managers, support analysts, and compliance officers—by enabling natural language querying of real customer pain points across financial services. The project includes:
- A robust and modular data preprocessing pipeline.
- Exploratory Data Analysis (EDA) to inspect the quality and structure of customer complaints.
- A vector-based semantic retrieval pipeline powered by FAISS and Sentence Transformers.
- Prepared data and embeddings to support Retrieval-Augmented Generation (RAG) chatbot logic.

# Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/matos-coder/Intelligent-Complaint-Analysis-for-Financial-Services
   cd Intelligent-Complaint-Analysis-for-Financial-Services
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate           # For Windows
   # OR
   source venv/bin/activate        # For Linux/macOS
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

# Project Structure

```
intelligent-complaint-analysis/
├── data/
│   └── filtered_complaints.csv     # Preprocessed dataset
├── notebooks/
│   └── EDA.ipynb                   # Initial data analysis and insights
├── src/
│   └── chunk_embed_index.py        # Chunking, embedding, and vector store creation
├── vector_store/
│   ├── faiss_index                 # FAISS vector index
│   └── metadata.pkl                # Metadata (chunk → source link)
├── requirements.txt
├── README.md
```

# Tasks

## Task 1: Exploratory Data Analysis (EDA) and Preprocessing

**Objective:**  
Understand the structure, content, and quality of the complaint data, and clean it for downstream semantic search and LLM processing.

**Location:**  
All analysis is implemented in `notebooks/EDA.ipynb`.

**Key Steps Performed:**
- **Data Inspection:**  
  - Dataset loaded from the Consumer Financial Protection Bureau (CFPB).
  - Inspected data types, null values, and initial statistics.
  - Counted number of complaints across financial products.
- **Narrative Length Analysis:**  
  - Calculated word counts for each Consumer complaint narrative.
  - Visualized distribution of complaint lengths to identify very short/long cases.
- **Filtering:**  
  - Focused on 5 relevant financial products:  
    - Credit card  
    - Personal loan  
    - Buy Now, Pay Later (BNPL)  
    - Savings account  
    - Money transfers  
  - Removed complaints with missing narratives.
- **Text Cleaning:**  
  - Lowercased all narratives.
  - Removed special characters and boilerplate legal language (e.g., "I am writing to file a complaint...").
  - Stripped whitespace for consistency.
- **Saving Cleaned Dataset:**  
  - Final cleaned and filtered dataset saved as `data/processed/filtered_complaints.csv`.

**Key Insights from EDA:**
- Original dataset had {total_complaints} complaints (auto-filled when run).
- After filtering, {filtered_count} complaints remained, all related to target financial products.
- Long-tail distribution of complaint lengths observed—some very verbose narratives.
- Missing narrative field found in ~{missing_count} records and excluded from training.
- Most complaints were about BNPL and Credit Cards, indicating high user dissatisfaction in those areas.

## Task 2: Text Chunking, Embedding, and Vector Indexing

**Objective:**  
Convert the cleaned complaints into dense vector representations suitable for efficient semantic search using Retrieval-Augmented Generation (RAG).

**Location:**  
The entire pipeline is implemented in `src/chunk_embed_index.py`.

**Components and Workflow:**
- **Chunking:**  
  - Used LangChain’s RecursiveCharacterTextSplitter to split long narratives into smaller, coherent text chunks.
  - Applied a chunk_size = 800 and chunk_overlap = 150 for optimal balance between semantic completeness and context retention.
- **Embedding:**  
  - Used the pre-trained `sentence-transformers/all-MiniLM-L6-v2` model.
  - Efficient, accurate, and supports local CPU-based embedding generation.
- **Metadata Mapping:**  
  - Stored metadata with each chunk:  
    - Complaint ID  
    - Product category  
    - Chunk index  
    - Original chunk text  
  - This supports explainability in the chatbot output.
- **Vector Store Indexing:**  
  - Created a dense vector index using Facebook’s FAISS.
  - Saved both the FAISS index and metadata to the `vector_store/` directory for reuse in the chatbot backend.

**Justifications for Technical Choices:**
- **Chunk Size:**  
  - Chose a chunk size of 800 characters with 150 overlap to ensure that each chunk retains full semantic meaning, minimizing risk of context loss across splits.
- **Embedding Model:**  
  - `all-MiniLM-L6-v2` was selected for its excellent trade-off between speed and semantic accuracy, while being lightweight enough for local development.

**Output:**
- After successful execution, the following files are generated:
  - `vector_store/faiss_index`: Dense vector index for fast similarity search.
  - `vector_store/metadata.pkl`: Metadata dictionary mapping vectors to complaint info.

# Project Goals

- Enable instant, natural-language querying of customer complaints.
- Reduce manual analysis time from hours to minutes.
- Support multiple product teams with a unified AI interface.
- Lay the foundation for building a Retrieval-Augmented Generation chatbot (Task 3 & 4).

# How to Run the Pipeline

## Task 1 - Data Preprocessing

- Run inside a Jupyter Notebook:
  ```
  notebooks/EDA.ipynb
  ```

## Task 2 - Chunking & Indexing

- From the project root, run:
  ```bash
  python src/chunk_embed_index.py
  ```

# Requirements

Use the following to install all required packages:
```bash
pip install -r requirements.txt
```
**Content includes:**
- nginx
- pandas
- numpy
- tqdm
- langchain
- sentence-transformers
- faiss-cpu

# Contact

For questions, feedback, or contributions, feel free to open an issue or submit a pull request.