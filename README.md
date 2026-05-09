# UAE Labour Law Intelligent Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about UAE Labour Law using hybrid retrieval (semantic search + BM25 keyword search) and LLM-based generation.

## Features

- Hybrid retrieval combining semantic embeddings and BM25 keyword search
- Grounded answers citing official UAE labour documents
- Interactive Streamlit web interface
- Persistent ChromaDB vector database

## System Architecture

- Data Ingestion (ingest.py): Load PDFs, chunk text, create embeddings
- Hybrid Retrieval (retrieval.py): Semantic + BM25 search combined with re-ranking
- Generation (generate.py): Prompt engineering + Claude LLM
- Web Interface (app.py): Streamlit dashboard

## Installation

1. Clone: git clone https://github.com/[yourname]/uae-labour-rag.git
2. Create venv: python -m venv venv
3. Activate: source venv/bin/activate (or venv\Scripts\activate on Windows)
4. Install: pip install -r requirements.txt

## Setup

1. Get API key: Sign up at https://console.anthropic.com
2. Export: export ANTHROPIC_API_KEY='your-key'
3. Download PDFs to data/raw/
4. Run: python src/ingest.py

## Usage

streamlit run src/app.py
Then open: http://localhost:8501

## Project Structure

- data/raw/ — Downloaded PDFs
- data/processed/ — Cleaned text
- src/ingest.py — Data ingestion & embedding
- src/retrieval.py — Hybrid retrieval
- src/generate.py — LLM generation
- src/app.py — Streamlit interface

## Team Members

- Alia: Data collection and problem definition
- Salma: Data cleaning and preprocessing
- Rana: Chunking, embedding, and ChromaDB setup
- Aya: Hybrid retrieval and LLM generation
- Reem: Streamlit web interface

## Technologies

Python 3.9+, LangChain, Sentence Transformers, ChromaDB, Rank-BM25, Claude API, Streamlit

## Evaluation Metrics

- Retrieval Relevance: Are retrieved chunks relevant?
- Answer Correctness: Is the answer accurate?
- Coverage: Can system answer diverse questions?
- Hallucination Rate: How often does it make up info?

## License

CSAI-413 Course Project 
