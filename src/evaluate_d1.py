import os
import sys
import csv
import time
from pathlib import Path

# =========================================================
# FIND PROJECT ROOT SAFELY
# =========================================================

CURRENT_FILE_DIR = Path(__file__).resolve().parent

# If this file is inside src/
if CURRENT_FILE_DIR.name == "src":
    PROJECT_ROOT = CURRENT_FILE_DIR.parent
else:
    PROJECT_ROOT = CURRENT_FILE_DIR

SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
os.chdir(PROJECT_ROOT)

print(f"✅ Project root: {PROJECT_ROOT}")
print(f"✅ Source folder: {SRC_DIR}")


# =========================================================
# LOAD .ENV
# =========================================================

from dotenv import load_dotenv

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found.")
    print(f"Expected .env location: {ENV_PATH}")
    print("Add this inside .env:")
    print("GEMINI_API_KEY=your-key-here")
    sys.exit(1)

print("✅ GEMINI_API_KEY found")


# =========================================================
# IMPORT PIPELINE
# =========================================================

from generate import answer_question


# =========================================================
# D1 INITIAL EVALUATION QUESTIONS
# =========================================================

questions = [
    "What is the minimum annual leave in the UAE?",
    "How is end of service gratuity calculated?",
    "What are the rules for probation period?",
    "What is the maximum working hours per week?",
    "How can an employee file a labour complaint?",
    "Can an employer terminate an employee without notice?",
    "What is the Wage Protection System?",
    "What are maternity leave rights in the UAE?",
    "What are the rules for remote work in the UAE?",
    "Can an employer reduce salary without employee consent?"
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def short_text(text, max_chars=500):
    """
    Shorten long text so the CSV stays readable.
    """
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def sources_summary(context_chunks):
    """
    Create a readable list of retrieved sources and scores.
    """
    lines = []

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("hybrid_score", "N/A")
        chunk_index = chunk.get("chunk_index", "N/A")

        lines.append(
            f"{i}. {source} | chunk {chunk_index} | score {score}"
        )

    return "\n".join(lines)


def chunks_preview(context_chunks):
    """
    Create a short preview of retrieved chunks for evaluation.
    """
    lines = []

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("hybrid_score", "N/A")
        text = short_text(chunk.get("text", ""), max_chars=350)

        lines.append(
            f"[Chunk {i}] Source: {source} | Score: {score}\n{text}"
        )

    return "\n\n".join(lines)


# =========================================================
# RUN EVALUATION
# =========================================================

rows = []

print("\n" + "=" * 70)
print("D1 INITIAL EVALUATION")
print("=" * 70)

for i, question in enumerate(questions, 1):
    print(f"\nQ{i}: {question}")
    print("-" * 70)

    try:
        result = answer_question(question)

        answer = result.get("answer", "")
        context_chunks = result.get("context_chunks", [])

        if context_chunks:
            top_chunk = context_chunks[0]
            top_source = top_chunk.get("source", "unknown")
            top_score = top_chunk.get("hybrid_score", "N/A")
        else:
            top_source = "No retrieved source"
            top_score = "N/A"

        row = {
            "Question No.": i,
            "Question": question,
            "Top Retrieved Source": top_source,
            "Top Retrieval Score": top_score,
            "All Retrieved Sources": sources_summary(context_chunks),
            "Retrieved Chunk Preview": chunks_preview(context_chunks),
            "Generated Answer": answer,
            "Correct?": "",          # Fill manually: Yes / Partially / No
            "Issues / Notes": ""     # Fill manually after checking
        }

        rows.append(row)

        print(f"Top Source: {top_source}")
        print(f"Top Score: {top_score}")
        print("\nGenerated Answer:")
        print(answer)

    except Exception as e:
        row = {
            "Question No.": i,
            "Question": question,
            "Top Retrieved Source": "ERROR",
            "Top Retrieval Score": "ERROR",
            "All Retrieved Sources": "ERROR",
            "Retrieved Chunk Preview": "ERROR",
            "Generated Answer": f"ERROR: {e}",
            "Correct?": "No",
            "Issues / Notes": f"Pipeline failed: {e}"
        }

        rows.append(row)

        print(f"❌ Error: {e}")

    print("-" * 70)

    # Avoid sending Gemini requests too quickly
    time.sleep(8)


# =========================================================
# SAVE CSV FILE
# =========================================================

output_csv = REPORTS_DIR / "d1_initial_evaluation_results.csv"

fieldnames = [
    "Question No.",
    "Question",
    "Top Retrieved Source",
    "Top Retrieval Score",
    "All Retrieved Sources",
    "Retrieved Chunk Preview",
    "Generated Answer",
    "Correct?",
    "Issues / Notes"
]

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n✅ D1 evaluation complete.")
print(f"✅ Results saved to: {output_csv}")
print("\nOpen this CSV and fill:")
print("- Correct? = Yes / Partially / No")
print("- Issues / Notes = None, weak retrieval, incomplete answer, wrong source, etc.")