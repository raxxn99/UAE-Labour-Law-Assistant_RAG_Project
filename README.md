# UAE Labour Law Intelligent Assistant

An advanced, enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to resolve complex statutory compliance queries regarding UAE Labour Law. The pipeline implements hybrid information retrieval—combining dense semantic vector embeddings with sparse BM25 keyword matching—paired with Google's Gemini 2.5 Flash LLM for highly grounded, hallucination-resistant legal inference.

---

## 🚀 Key Features

* **Hybrid Retrieval Architecture:** Executes dual-channel matrix matching using dense vector space mapping alongside traditional lexical token indices.
* **Deterministic Legal Grounding:** Mitigates hallucination boundaries by hard-restricting text synthesis strictly to validated source references.
* **Modern Premium User Interface:** Features a responsive, dark-mode Streamlit workspace built with analytical insight tabs and execution metrics.
* **Persistent Vector Framework:** Ships with a pre-computed vector space comprising 2,033 legal document fractions.

---

## 📊 System Architecture

| Operational Phase | Source Module | Technical Profile & Intent |
|:---|:---|:---|
| **Data Normalization** | `src/data_cleaning.py` | Structural preprocessing, metadata extraction, and cleansing of raw source material. |
| **Ingestion Pipeline** | `src/ingest.py` | Implements token-bounded text segmentation, vector embedding generation, and DB seeding. |
| **Hybrid Retrieval** | `src/retrieval.py` | Executes parallel semantic + BM25 indexing, normalized via Reciprocal Rank Fusion ($RRF$). |
| **Contextual Synthesis** | `src/generate.py` | Strict parameter prompt engineering utilizing the Google Gemini 2.5 Flash API layer. |
| **Web Interface** | `app.py` | Interactive dashboard layout presenting conversation logs and context tracking. |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/raxxn99/UAE-Labour-Law-Assistant_RAG_Project.git](https://github.com/raxxn99/UAE-Labour-Law-Assistant_RAG_Project.git)
cd UAE-Labour-Law-Assistant_RAG_Project

```

### 2. Configure Virtual Environment

```bash
# Initialize environment
python -m venv venv

# Activation (Windows Command Prompt)
venv\Scripts\activate

# Activation (Mac / Linux Terminal)
source venv/bin/activate

```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt

```

---

## 🔑 API Key Setup (Required for Live Evaluation)

This system features an automated **Inference Simulation Mode** if credentials are missing. To activate the live RAG generation pipeline, you must supply a free Google AI Studio token:

1. Generate your API credentials via [Google AI Studio](https://aistudio.google.com/apikey).
2. Go to a `.env` in the absolute root directory of this project.
3. Paste your credentials exactly as configured below:

```env
GEMINI_API_KEY=your_actual_copied_api_token_here

```

> ⚠️ **Security Enforcement:** The `.env` profile is explicitly registered within `.gitignore` and will never be committed to shared public version control logs.

---

## 🏃 Execution Manual

### Step 1 — Pre-processed Assets

The baseline legal documents are optimized and stored as normalized structures inside `data/processed_data/`.

### Step 2 — Vector Database Status

The underlying `chroma_db/` persistent storage directory contains 2,033 pre-indexed data fragments. Running `src/ingest.py` is **not required** unless you are introducing additional legal text documents to the corpus.

### Step 3 — Run Integration Verification (Recommended)

Before launching the user interface, verify underlying pipeline communication limits by executing the automated validation harness:

```bash
python test_Pipeline.py

```

### Step 4 — Launch the Web Portal

Execute the unified script through the Streamlit engine:

```bash
streamlit run app.py

```

Once initialized, navigate your local browser instances directly to: **http://localhost:8501**

---

## 📁 Repository Structure

```
UAE-LABOUR-LAW-ASSISTANT_RAG_PROJECT/
├── chroma_db/                  — Persistent vector database storage (2,033 pre-indexed chunks)
├── data/
│   ├── raw_data/               — Source legal decrees and original documentation
│   └── processed_data/         — Formatted and parsed plain-text records
├── src/
│   ├── data_cleaning.py        — Scripting logic for structural content cleaning
│   ├── ingest.py               — Splitting, text chunking, embedding, and vector database generation
│   ├── retrieval.py            — Hybrid evaluation (Parallel Semantic Vector Distance + BM25 Tracking)
│   └── generate.py             — Prompt isolation structure routing to the Gemini LLM
├── .env                        — Local configuration environment file (User Defined)
├── .gitignore
├── app.py                      — Streamlit interactive dashboard script (Root Level execution)
├── requirements.txt            — Absolute library versions manifest
└── test_full.py                — Non-interactive automation validation script

```

---

## 🎛️ Technology Stack Profile

* **Core Platform Engine:** Python 3.11
* **Segmentation Protocols:** LangChain Token-Bounded Recursive Text Splitters
* **Dense Embedding Matrix:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Index Ecosystem:** ChromaDB Vector Storage
* **Sparse Index Optimization:** Rank-BM25 Lexical Search Frame
* **Inference Layer Engine:** Google Gemini 2.5 Flash
* **User Delivery Portal:** Streamlit Pro Dashboard Suite

---

## 📊 Core Performance Evaluation Metrics

To guarantee robust legal alignment and application safety, execution layers are explicitly scored against four fundamental metrics:

* **Retrieval Relevance:** Precision assessment of chunk arrays extracted through the hybrid scoring framework ($0.7 \times \text{Semantic} + 0.3 \times \text{BM25}$).
* **Answer Correctness:** Veracity indexing of compiled responses evaluated directly against matching statutory decree guidelines.
* **Domain Coverage:** Dimensional evaluation asserting the system's resilience across highly diverse operational labor scenarios.
* **Hallucination Suppression:** Quantitative evaluation confirming zero-tolerance metrics for target assertions unsupported by the reference database context window.

---

### 🏛️ Academic Course Context

**CSAI-413: Natural Language Processing Applications**

Faculty of Engineering and IT · The British University in Dubai · 2026

```

```