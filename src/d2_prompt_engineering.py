import os
import sys
import csv
import time
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from google import genai

# =========================================================
# PATH SETUP
# =========================================================

CURRENT_FILE_DIR = Path(__file__).resolve().parent

if CURRENT_FILE_DIR.name == "src":
    PROJECT_ROOT = CURRENT_FILE_DIR.parent
else:
    PROJECT_ROOT = CURRENT_FILE_DIR

SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
os.chdir(PROJECT_ROOT)

from retrieval import retrieve


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv(PROJECT_ROOT / ".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file in the project root with:\n"
        "GEMINI_API_KEY=your_key_here"
    )


# =========================================================
# CONFIGURATION
# =========================================================

TOP_K = 5

LLM_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

MAX_RETRIES_PER_MODEL = 3
RETRY_WAIT_SECONDS = 8
DELAY_BETWEEN_CALLS = 6


# =========================================================
# PROMPT TEMPLATES FOR D2
# =========================================================

PROMPT_TEMPLATES = {
    "Original": """
You are an intelligent assistant specialized in UAE Labour Law.

Answer the user's question using ONLY the context provided below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not clearly found in the context, say:
   "I cannot find this information in the available UAE Labour Law documents."
3. Give a clear and direct answer.
4. Mention the source document names when possible.
5. Do not invent article numbers, legal rules, dates, or exceptions.
6. Keep the answer understandable for employees, HR staff, and students.

Context:
{context}

Question:
{question}

Answer:
""",

    "Strict": """
You are a UAE Labour Law RAG assistant.

You must answer using ONLY the provided context.
Do not use general legal knowledge.
Do not guess.
Do not add facts that are not explicitly supported by the context.

If the context does not contain enough evidence, answer exactly:
"The retrieved context does not provide enough information to answer this question."

Your answer must:
- Be concise.
- Avoid unsupported explanations.
- Mention only facts directly found in the retrieved chunks.

Context:
{context}

Question:
{question}

Answer:
""",

    "Citational": """
You are a UAE Labour Law assistant.

Answer the question using ONLY the retrieved context.
Every important claim must be linked to a source document name.
Use this style:
- Answer: ...
- Evidence: Source document name and chunk number.

Do not invent article numbers, legal exceptions, or procedures.
If the retrieved context is insufficient, clearly state that the evidence is insufficient.

Context:
{context}

Question:
{question}

Answer with citations:
""",

    "Simple": """
Explain the answer in simple language for a normal employee or HR staff member.

Use only the information in the context.
Do not use outside knowledge.
If the answer is not in the context, say that the available documents do not provide enough information.

Context:
{context}

Question:
{question}

Simple answer:
"""
}


# =========================================================
# TEST QUESTIONS
# =========================================================

QUESTIONS = [
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

def format_context(context_chunks: List[Dict[str, Any]]) -> str:
    context_text = ""

    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"[Chunk {i}]\n"
        context_text += f"Source: {chunk.get('source', 'unknown')}\n"
        context_text += f"Chunk Index: {chunk.get('chunk_index', 'unknown')}\n"
        context_text += f"Hybrid Score: {chunk.get('hybrid_score', 0)}\n"
        context_text += f"Text:\n{chunk.get('text', '')}\n\n"

    return context_text.strip()


def sources_summary(context_chunks: List[Dict[str, Any]]) -> str:
    lines = []

    for i, chunk in enumerate(context_chunks, 1):
        lines.append(
            f"{i}. {chunk.get('source', 'unknown')} "
            f"| chunk {chunk.get('chunk_index', 'unknown')} "
            f"| score {chunk.get('hybrid_score', 0)}"
        )

    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(str(text).split())


def has_citation(answer: str) -> str:
    answer_lower = answer.lower()

    if "source" in answer_lower or "evidence" in answer_lower or ".txt" in answer_lower or "chunk" in answer_lower:
        return "Yes"

    return "No"


def detects_insufficient_context(answer: str) -> str:
    answer_lower = answer.lower()

    phrases = [
        "not provide enough information",
        "cannot find",
        "insufficient",
        "not clearly found",
        "available documents do not provide"
    ]

    for phrase in phrases:
        if phrase in answer_lower:
            return "Yes"

    return "No"


def generate_with_prompt(question: str, context_chunks: List[Dict[str, Any]], prompt_template: str):
    context = format_context(context_chunks)

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = None

    for model_name in LLM_MODELS:
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                print(f"Trying {model_name} | Attempt {attempt}")

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                return response.text, model_name, ""

            except Exception as e:
                last_error = e
                error_text = str(e)

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):
                    print(f"Gemini busy. Waiting {RETRY_WAIT_SECONDS} seconds...")
                    time.sleep(RETRY_WAIT_SECONDS)
                    continue

                return "", model_name, str(e)

    return "", "All models failed", str(last_error)


# =========================================================
# RUN D2 PROMPT EXPERIMENT
# =========================================================

rows = []

print("\n" + "=" * 70)
print("D2 PROMPT ENGINEERING EXPERIMENT")
print("=" * 70)

for q_no, question in enumerate(QUESTIONS, 1):
    print(f"\nRetrieving context for Q{q_no}: {question}")

    context_chunks = retrieve(question, top_k=TOP_K)

    top_source = context_chunks[0].get("source", "unknown") if context_chunks else "No source"
    top_score = context_chunks[0].get("hybrid_score", 0) if context_chunks else 0
    all_sources = sources_summary(context_chunks)

    for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
        print(f"\nQ{q_no} | Prompt: {prompt_name}")
        print("-" * 70)

        answer, used_model, error = generate_with_prompt(
            question=question,
            context_chunks=context_chunks,
            prompt_template=prompt_template
        )

        row = {
            "Question No.": q_no,
            "Question": question,
            "Prompt Type": prompt_name,
            "Model Used": used_model,
            "Top Retrieved Source": top_source,
            "Top Retrieval Score": top_score,
            "All Retrieved Sources": all_sources,
            "Generated Answer": answer,
            "Answer Word Count": word_count(answer),
            "Mentions Source/Citation": has_citation(answer),
            "Detects Insufficient Context": detects_insufficient_context(answer),
            "API Error": error,

            # Fill these manually after checking the answer.
            "Manual Relevance Score (1-5)": "",
            "Manual Correctness Score (1-5)": "",
            "Manual Grounding Score (1-5)": "",
            "Manual Hallucination Score (1-5, lower is better)": "",
            "Manual Notes / Issues": ""
        }

        rows.append(row)

        print(answer[:600])
        print("-" * 70)

        time.sleep(DELAY_BETWEEN_CALLS)


# =========================================================
# SAVE RESULTS
# =========================================================

output_file = TABLES_DIR / "d2_prompt_engineering_results.csv"

fieldnames = list(rows[0].keys())

with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n✅ D2 prompt engineering experiment complete.")
print(f"✅ Results saved to: {output_file}")
print("\nNext step:")
print("Open the CSV and manually fill the scoring columns.")