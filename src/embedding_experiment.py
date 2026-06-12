"""
src/embedding_experiment.py
Salma — D2: Embedding Model Comparison Experiment

Compares three SentenceTransformer embedding models on retrieval quality
and speed against 10 UAE Labour Law evaluation questions.

Each model is stored in its own isolated ChromaDB collection so that
no model's data can ever contaminate another's results.

Outputs (written to reports/):
    salma_d2_detailed_results.csv  — one row per (model × question)
    salma_d2_summary.csv           — one row per model (aggregate metrics)

Run from the project root:
    python src/embedding_experiment.py
"""

import os
import sys
import csv
import time
import statistics
from pathlib import Path
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP — works whether launched from project root or from src/
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "src" else _THIS_DIR
REPORTS_DIR  = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
os.chdir(PROJECT_ROOT)

print(f"✅ Project root  : {PROJECT_ROOT}")
print(f"✅ Reports folder: {REPORTS_DIR}")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Three models chosen at different size / quality points:
#   1. all-MiniLM-L6-v2         — current production baseline (fast, 384-dim)
#   2. all-mpnet-base-v2        — larger capacity, richer 768-dim vectors
#   3. multi-qa-MiniLM-L6-cos-v1 — fine-tuned specifically on Q&A datasets;
#                                   expected to suit legal question retrieval best

MODELS: List[Dict] = [
    {
        "model_name": "all-MiniLM-L6-v2",
        "collection": "exp_minilm_l6",
        "label":      "MiniLM-L6",
        "dims":       384,
        "description": "Baseline — lightweight 384-dim general-purpose model (~22 MB)",
    },
    {
        "model_name": "all-mpnet-base-v2",
        "collection": "exp_mpnet_base",
        "label":      "MPNet-Base",
        "dims":       768,
        "description": "High-quality — 768-dim, richer semantic capacity (~420 MB)",
    },
    {
        "model_name": "multi-qa-MiniLM-L6-cos-v1",
        "collection": "exp_qa_minilm",
        "label":      "QA-MiniLM",
        "dims":       384,
        "description": "QA-optimised — fine-tuned on question-answer pairs, 384-dim (~22 MB)",
    },
]

# Production chunking settings kept constant so chunking is not a variable
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
DATA_DIR      = str(PROJECT_ROOT / "data" / "processed_data")
CHROMA_PATH   = str(PROJECT_ROOT / "chroma_db")
TOP_K         = 5

# 10 diverse questions covering different UAE Labour Law topic areas
EVAL_QUESTIONS: List[str] = [
    "What is the minimum annual leave entitlement in the UAE?",
    "How is end of service gratuity calculated?",
    "What are the rules for an employee probation period?",
    "What is the maximum number of working hours per week?",
    "Can an employer dismiss an employee without any notice?",
    "What are the maternity leave rights for female employees?",
    "How can an employee submit a labour complaint in the UAE?",
    "What is the Wage Protection System and how does it work?",
    "Are non-compete clauses legally enforceable in UAE contracts?",
    "What happens to unused annual leave if an employee resigns?",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND CHUNK DOCUMENTS
# Chunking is done once and shared across all models for a fair comparison.
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk(data_dir: str) -> List[Dict]:
    print(f"\n{'─'*60}")
    print(f"  LOADING DOCUMENTS FROM: {data_dir}")
    print(f"{'─'*60}")

    if not os.path.isdir(data_dir):
        raise RuntimeError(
            f"Data directory not found: {data_dir}\n"
            "Run src/data_cleaning.py first to generate processed text files."
        )

    documents = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt"):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                documents.append({"content": f.read(), "source": fname})

    if not documents:
        raise RuntimeError(f"No .txt files found in {data_dir}.")

    print(f"  Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc["content"])):
            chunks.append({
                "text":     text,
                "source":   doc["source"],
                "chunk_id": f"{doc['source']}_chunk_{i}",
            })

    print(f"  Created {len(chunks)} chunks  "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — INGEST ONE MODEL INTO ITS OWN CHROMADB COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def ingest_model(model_cfg: Dict, chunks: List[Dict]) -> Dict:
    """
    Load the model, encode every chunk, and upsert into a dedicated collection.
    Returns the live model object, the collection handle, and timing metrics.
    """
    print(f"\n{'='*60}")
    print(f"  INGESTING: {model_cfg['model_name']}")
    print(f"  Collection: {model_cfg['collection']}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_cfg["model_name"])

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete stale collection from a previous run so scores are always fresh
    try:
        client.delete_collection(model_cfg["collection"])
        print("  Deleted existing collection — starting clean")
    except Exception:
        pass

    collection = client.create_collection(model_cfg["collection"])

    texts = [c["text"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]
    metas = [
        {"source": c["source"], "chunk_index": str(i)}
        for i, c in enumerate(chunks)
    ]

    print(f"  Encoding {len(texts)} chunks with {model_cfg['model_name']} …")
    t_start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64).tolist()
    ingestion_time = time.time() - t_start

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)

    ms_per_chunk = (ingestion_time / len(texts)) * 1000
    print(f"  ✅ Ingested {len(texts)} chunks in {ingestion_time:.2f}s "
          f"({ms_per_chunk:.2f} ms/chunk)")

    return {
        "model":         model,
        "collection":    collection,
        "ingestion_s":   round(ingestion_time, 2),
        "ms_per_chunk":  round(ms_per_chunk, 3),
        "total_chunks":  len(texts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EVALUATE ONE MODEL ON ALL QUESTIONS
# Pure semantic search only — BM25 is excluded because it does not depend on
# the embedding model and would make scores identical across all models.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_cfg: Dict,
    model: SentenceTransformer,
    collection,
    questions: List[str],
) -> List[Dict]:
    """
    For each question: encode the query, search ChromaDB, convert distances
    to similarity scores, and record all timing data.
    """
    print(f"\n  Evaluating '{model_cfg['label']}' on {len(questions)} questions …")
    results = []

    for q_num, question in enumerate(questions, 1):
        # Measure query encoding time separately from ChromaDB search time
        t0 = time.time()
        q_vec = model.encode(question).tolist()
        encode_ms = round((time.time() - t0) * 1000, 2)

        t0 = time.time()
        raw = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
            include=["distances"],
        )
        retrieval_ms = round((time.time() - t0) * 1000, 2)

        distances = raw["distances"][0]

        # Same distance-to-similarity formula used by production retrieval.py
        # so scores are directly comparable to real system behaviour
        sims = [round(1.0 / (1.0 + d), 4) for d in distances]

        # Pad to TOP_K if fewer chunks were returned
        while len(sims) < TOP_K:
            sims.append(0.0)

        avg  = round(statistics.mean(sims), 4)
        top1 = sims[0]

        results.append({
            "model_name":   model_cfg["model_name"],
            "label":        model_cfg["label"],
            "q_num":        q_num,
            "question":     question,
            "top_1":        sims[0],
            "top_2":        sims[1],
            "top_3":        sims[2],
            "top_4":        sims[3],
            "top_5":        sims[4],
            "avg_score":    avg,
            "encode_ms":    encode_ms,
            "retrieval_ms": retrieval_ms,
        })

        print(f"    Q{q_num:02d} | avg={avg:.4f}  top-1={top1:.4f}  "
              f"enc={encode_ms}ms  ret={retrieval_ms}ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — AGGREGATE PER-MODEL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(
    model_cfg: Dict,
    ingest_metrics: Dict,
    question_results: List[Dict],
) -> Dict:
    avg_scores     = [r["avg_score"]    for r in question_results]
    top1_scores    = [r["top_1"]        for r in question_results]
    encode_times   = [r["encode_ms"]    for r in question_results]
    retrieval_times= [r["retrieval_ms"] for r in question_results]

    return {
        "model_name":         model_cfg["model_name"],
        "label":              model_cfg["label"],
        "dims":               model_cfg["dims"],
        "description":        model_cfg["description"],
        "total_chunks":       ingest_metrics["total_chunks"],
        "ingestion_time_s":   ingest_metrics["ingestion_s"],
        "ms_per_chunk":       ingest_metrics["ms_per_chunk"],
        "mean_avg_score":     round(statistics.mean(avg_scores), 4),
        "mean_top1_score":    round(statistics.mean(top1_scores), 4),
        "stdev_avg_score":    round(statistics.stdev(avg_scores), 4) if len(avg_scores) > 1 else 0.0,
        "min_avg_score":      round(min(avg_scores), 4),
        "max_avg_score":      round(max(avg_scores), 4),
        "mean_encode_ms":     round(statistics.mean(encode_times), 2),
        "mean_retrieval_ms":  round(statistics.mean(retrieval_times), 2),
        "total_eval_questions": len(question_results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_detailed_csv(all_results: List[Dict]) -> Path:
    output_path = REPORTS_DIR / "salma_d2_detailed_results.csv"
    fieldnames = [
        "model_name", "label", "q_num", "question",
        "top_1", "top_2", "top_3", "top_4", "top_5",
        "avg_score", "encode_ms", "retrieval_ms",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✅ Detailed results saved  → {output_path}")
    return output_path


def save_summary_csv(summaries: List[Dict]) -> Path:
    output_path = REPORTS_DIR / "salma_d2_summary.csv"
    fieldnames = [
        "model_name", "label", "dims", "description",
        "total_chunks", "ingestion_time_s", "ms_per_chunk",
        "mean_avg_score", "mean_top1_score", "stdev_avg_score",
        "min_avg_score", "max_avg_score",
        "mean_encode_ms", "mean_retrieval_ms",
        "total_eval_questions",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"✅ Summary table saved      → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PRINT FORMATTED COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(summaries: List[Dict]) -> None:
    print("\n" + "=" * 90)
    print("  SALMA D2 — EMBEDDING MODEL COMPARISON RESULTS")
    print("=" * 90)

    header = (
        f"  {'Model':<16} {'Dims':>5}  "
        f"{'AvgScore':>9}  {'Top-1':>7}  {'StdDev':>7}  "
        f"{'Ingest(s)':>10}  {'Enc(ms)':>8}  {'Ret(ms)':>8}"
    )
    print(header)
    print("  " + "-" * 86)

    best = max(summaries, key=lambda x: x["mean_avg_score"])

    for s in summaries:
        star = " ◀ BEST RETRIEVAL" if s["label"] == best["label"] else ""
        row = (
            f"  {s['label']:<16} {s['dims']:>5}  "
            f"{s['mean_avg_score']:>9.4f}  "
            f"{s['mean_top1_score']:>7.4f}  "
            f"{s['stdev_avg_score']:>7.4f}  "
            f"{s['ingestion_time_s']:>10.2f}  "
            f"{s['mean_encode_ms']:>8.2f}  "
            f"{s['mean_retrieval_ms']:>8.2f}"
            f"{star}"
        )
        print(row)

    print("  " + "-" * 86)
    print(f"\n  Winner (retrieval quality): {best['model_name']}")
    print(f"  Description              : {best['description']}")
    print("=" * 90)

    # Per-question breakdown
    print("\n  COLUMN DEFINITIONS")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  AvgScore   = mean similarity score of top-5 retrieved chunks")
    print("  Top-1      = similarity of the single best-matching chunk")
    print("  StdDev     = score standard deviation across 10 questions (lower = more consistent)")
    print("  Ingest(s)  = total time (seconds) to encode all chunks at ingestion")
    print("  Enc(ms)    = average time to encode one query (milliseconds)")
    print("  Ret(ms)    = average ChromaDB vector search time (milliseconds)")
    print("  Similarity = 1 / (1 + L2_distance)  → higher is better, range (0, 1]")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  SALMA D2 — EMBEDDING MODEL COMPARISON EXPERIMENT")
    print("=" * 60)
    print(f"  Data dir       : {DATA_DIR}")
    print(f"  Models tested  : {len(MODELS)}")
    print(f"  Eval questions : {len(EVAL_QUESTIONS)}")
    print(f"  Chunk size     : {CHUNK_SIZE}  (overlap: {CHUNK_OVERLAP})")
    print(f"  Top-K          : {TOP_K}")

    # Load and chunk once — identical input for all models (fair comparison)
    chunks = load_and_chunk(DATA_DIR)

    all_question_results: List[Dict] = []
    summaries:            List[Dict] = []

    for model_cfg in MODELS:
        try:
            ingest_metrics   = ingest_model(model_cfg, chunks)
            question_results = evaluate_model(
                model_cfg,
                ingest_metrics["model"],
                ingest_metrics["collection"],
                EVAL_QUESTIONS,
            )
            summary = build_summary(model_cfg, ingest_metrics, question_results)

            all_question_results.extend(question_results)
            summaries.append(summary)

        except Exception as exc:
            print(f"\n❌ Failed on model '{model_cfg['model_name']}': {exc}")
            raise

    save_detailed_csv(all_question_results)
    save_summary_csv(summaries)
    print_comparison_table(summaries)

    print("Experiment complete.")
    print("Open reports/salma_d2_summary.csv and salma_d2_detailed_results.csv")
    print("for the full data to include in your D2 report.\n")


if __name__ == "__main__":
    main()
