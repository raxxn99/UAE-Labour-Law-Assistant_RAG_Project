import re
from typing import List, Dict, Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# =========================================================
# AYA D1 - HYBRID RETRIEVAL CONFIGURATION
# =========================================================

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "uae_labour_law"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


# =========================================================
# TEXT TOKENIZATION FOR BM25
# =========================================================

def tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase keyword tokens for BM25 search.
    This keeps words and numbers, and removes punctuation.
    """
    return re.findall(r"\b\w+\b", text.lower())


# =========================================================
# HYBRID RETRIEVER CLASS
# =========================================================

class HybridRetriever:
    """
    Hybrid retriever for the UAE Labour Law RAG system.

    It combines:
    1. Semantic retrieval using SentenceTransformer + ChromaDB
    2. Keyword retrieval using BM25
    3. Weighted re-ranking using both scores
    """

    def __init__(self):
        print("Loading ChromaDB collection...")

        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                f"Could not find ChromaDB collection '{COLLECTION_NAME}'.\n"
                f"Make sure Rana has already run: python src/ingest.py\n"
                f"And make sure the database exists in: {CHROMA_DB_PATH}/"
            )

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.bm25 = None

        self._build_bm25_index()

    def _build_bm25_index(self):
        """
        Load all stored chunks from ChromaDB and build BM25 keyword index.
        """
        all_docs = self.collection.get(include=["documents", "metadatas"])

        self.ids = all_docs.get("ids", [])
        self.documents = all_docs.get("documents", [])
        self.metadatas = all_docs.get("metadatas", [])

        if not self.documents:
            raise RuntimeError(
                "The ChromaDB collection is empty. "
                "Run Rana's ingest.py first after Salma creates processed .txt files."
            )

        tokenized_docs = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"BM25 keyword index built with {len(self.documents)} chunks.")

    def semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar chunks using ChromaDB vector search.
        """
        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        semantic_results = []

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
            # Chroma distance is lower when more similar.
            # Convert distance to similarity in a safe way.
            similarity = 1 / (1 + distance)

            semantic_results.append({
                "id": chunk_id,
                "text": doc,
                "metadata": meta,
                "semantic_score": float(similarity),
                "keyword_score": 0.0
            })

        return semantic_results

    def keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve keyword-relevant chunks using BM25.
        """
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[-top_k:][::-1]
        max_score = max(scores) if max(scores) > 0 else 1.0

        keyword_results = []

        for idx in top_indices:
            normalized_score = scores[idx] / max_score

            keyword_results.append({
                "id": self.ids[idx],
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "semantic_score": 0.0,
                "keyword_score": float(normalized_score)
            })

        return keyword_results

    def hybrid_retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Combine semantic search and BM25 keyword search using weighted scoring.
        """
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)

        combined = {}

        for result in semantic_results:
            combined[result["id"]] = result

        for result in keyword_results:
            chunk_id = result["id"]

            if chunk_id in combined:
                combined[chunk_id]["keyword_score"] = result["keyword_score"]
            else:
                combined[chunk_id] = result

        final_results = []

        for chunk_id, result in combined.items():
            hybrid_score = (
                SEMANTIC_WEIGHT * result["semantic_score"]
                + KEYWORD_WEIGHT * result["keyword_score"]
            )

            final_results.append({
                "id": chunk_id,
                "text": result["text"],
                "source": result["metadata"].get("source", "unknown"),
                "chunk_index": result["metadata"].get("chunk_index", "unknown"),
                "semantic_score": round(result["semantic_score"], 4),
                "keyword_score": round(result["keyword_score"], 4),
                "hybrid_score": round(float(hybrid_score), 4)
            })

        final_results = sorted(
            final_results,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return final_results[:top_k]


# =========================================================
# SIMPLE FUNCTION USED BY GENERATE.PY
# =========================================================

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Main retrieval function used by generate.py.
    """
    retriever = HybridRetriever()
    return retriever.hybrid_retrieve(query, top_k)


# =========================================================
# TEST RETRIEVAL DIRECTLY
# =========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the minimum annual leave in UAE?"

    print("\n=== TEST QUERY ===")
    print(question)

    results = retrieve(question, top_k=TOP_K)

    print("\n=== RETRIEVED CHUNKS ===")
    for i, chunk in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk['source']}")
        print(f"Chunk Index: {chunk['chunk_index']}")
        print(f"Semantic Score: {chunk['semantic_score']}")
        print(f"Keyword Score: {chunk['keyword_score']}")
        print(f"Hybrid Score: {chunk['hybrid_score']}")
        print(chunk["text"][:700])