# UAE Labour Law Intelligent Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about UAE Labour Law using hybrid retrieval (semantic search + BM25 keyword search) and Gemini LLM-based generation.

---

## Features

- Hybrid retrieval combining semantic embeddings and BM25 keyword search
- Grounded answers citing official UAE labour law documents
- Interactive Streamlit web interface
- Persistent ChromaDB vector database with 2,033 indexed chunks

---

## System Architecture

| Component | File | Description |
|-----------|------|-------------|
| Data Cleaning | `data_cleaning.py` | Clean and preprocess raw documents |
| Data Ingestion | `ingest.py` | Chunk text and create embeddings |
| Hybrid Retrieval | `retrieval.py` | Semantic + BM25 search with re-ranking |
| Generation | `generate.py` | Prompt engineering + Gemini 2.5 Flash LLM |
| Web Interface | `app.py` | Streamlit dashboard |

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/raxxn99/UAE-Labour-Law-Assistant_RAG_Project.git
```

**2. Create virtual environment:**
```bash
python -m venv venv
```

**3. Activate virtual environment:**
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## API Key Setup (Required)

This project uses the **Gemini API** (free).

1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **"Create API Key"** and copy it
4. in `.env` file in the project root and add:

```
GEMINI_API_KEY=your-key-here
```

> The `.env` file is listed in `.gitignore` and will never be pushed to GitHub.
> Every user must create their own free API key.

---

## Running the System

### Step 1 — Data is already processed
The `data/processed_data/` folder contains cleaned `.txt` files ready to use.

### Step 2 — ChromaDB is already built
The `chroma_db/` folder contains 2,033 pre-indexed chunks.
You do **NOT** need to run `ingest.py` again unless adding new documents.

### Step 3 — Test the pipeline (optional but recommended)
```bash
python test_full.py
```

If all checks pass, the backend is fully working.

### Step 4 — Run the app
```bash
streamlit run src/app.py
```

Then open: **http://localhost:8501**

---

## Project Structure

```
UAE-LABOUR-LAW-ASSISTANT_RAG_PROJECT/
├── chroma_db/                  — Persistent vector database (2,033 chunks)
├── data/
│   ├── raw_data/               — Original downloaded documents
│   └── processed_data/         — Cleaned .txt files ready for ingestion
├── src/
│   ├── data_cleaning.py        — Salma: data cleaning and preprocessing
│   ├── ingest.py               — Rana: chunking, embedding, ChromaDB
│   ├── retrieval.py            — Aya: hybrid retrieval (semantic + BM25)
│   ├── generate.py             — Aya: Gemini LLM generation
│   └── app.py                  — Reem: Streamlit web interface
├── .env                        — Your Gemini API key (NOT pushed to GitHub)
├── .gitignore
├── requirements.txt
├── test_full.py                — Full pipeline test script
└── README.md
```

---

## Team Members

| Member | Responsibility |
|--------|----------------|
| Alia   | Data collection, problem definition, and evaluation |
| Salma  | Data cleaning and preprocessing |
| Rana   | Chunking, embedding, and ChromaDB setup |
| Aya    | Hybrid retrieval and Gemini LLM generation |
| Reem   | Streamlit web interface |

---

## Technologies

| Technology | Purpose |
|------------|---------|
| Python 3.11 | Core language |
| LangChain Text Splitters | Document chunking |
| Sentence Transformers (all-MiniLM-L6-v2) | Embedding generation |
| ChromaDB | Persistent vector storage |
| Rank-BM25 | Keyword search |
| Google Gemini 2.5 Flash | LLM answer generation |
| Streamlit | Web interface |

---

## Evaluation Metrics

- **Retrieval Relevance** — Are retrieved chunks relevant to the question?
- **Answer Correctness** — Is the generated answer accurate?
- **Coverage** — Can the system answer diverse labour law questions?
- **Hallucination Rate** — How often does it generate unsupported information?

---

## License

CSAI-413 Course Project — The British University in Dubai
