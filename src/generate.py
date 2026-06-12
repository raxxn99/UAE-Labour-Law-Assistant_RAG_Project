import os
from typing import Dict, List, Any
import time
from dotenv import load_dotenv
from google import genai

from retrieval import retrieve


# =========================================================
# AYA D1 - GENERATION CONFIGURATION USING GEMINI
# =========================================================

load_dotenv()

LLM_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

MAX_RETRIES_PER_MODEL = 3
RETRY_WAIT_SECONDS = 8
TOP_K = 5

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# PROMPT TEMPLATE
# =========================================================

PROMPT_TEMPLATE = """
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
"""


# =========================================================
# CONTEXT FORMATTING
# =========================================================

def format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into a clean context block for Gemini.
    """
    context_text = ""

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        chunk_index = chunk.get("chunk_index", "unknown")
        score = chunk.get("hybrid_score", 0)

        context_text += f"[Chunk {i}]\n"
        context_text += f"Source: {source}\n"
        context_text += f"Chunk Index: {chunk_index}\n"
        context_text += f"Hybrid Score: {score}\n"
        context_text += f"Text:\n{chunk['text']}\n\n"

    return context_text.strip()


# =========================================================
# LLM GENERATION
# =========================================================

def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an answer using Gemini based on retrieved UAE Labour Law chunks.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing.\n"
            "Create a .env file in the project root and add:\n"
            "GEMINI_API_KEY=your_api_key_here"
        )

    context_text = format_context(context_chunks)

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )

    client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = None
    used_model = None
    answer = None

    for model_name in LLM_MODELS:
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                print(f"Trying Gemini model: {model_name} | Attempt {attempt}")

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                answer = response.text
                used_model = model_name
                break

            except Exception as e:
                last_error = e
                error_text = str(e)

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):
                    print(
                        f"⚠️ Gemini is busy for {model_name}. "
                        f"Waiting {RETRY_WAIT_SECONDS} seconds then retrying..."
                    )
                    time.sleep(RETRY_WAIT_SECONDS)
                    continue

                raise e

        if answer is not None:
            break

    if answer is None:
        raise RuntimeError(
            "All Gemini models failed because the API is currently busy. "
            "Please run the same question again after a few minutes.\n"
            f"Last error: {last_error}"
        )
    return {
        "question": question,
        "answer": answer,
        "context_chunks": context_chunks,
        "model": used_model,
        "top_k": TOP_K,
        "prompt_used": prompt
    }


# =========================================================
# FULL RAG PIPELINE
# =========================================================

def answer_question(question: str) -> Dict[str, Any]:
    """
    Complete Aya D1 RAG pipeline:
    1. Retrieve relevant chunks using hybrid retrieval
    2. Generate an answer using Gemini
    """
    print(f"Question: {question}")

    print("\nRetrieving relevant UAE Labour Law chunks...")
    retrieved_chunks = retrieve(question, top_k=TOP_K)
    print(f"Retrieved {len(retrieved_chunks)} chunks.")

    print("\nGenerating answer...")
    result = generate_answer(question, retrieved_chunks)

    return result


# =========================================================
# TEST GENERATION DIRECTLY
# =========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the minimum annual leave in UAE?"

    result = answer_question(question)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES USED ===")
    for i, chunk in enumerate(result["context_chunks"], 1):
        print(
            f"{i}. {chunk['source']} "
            f"| Chunk {chunk['chunk_index']} "
            f"| Hybrid Score: {chunk['hybrid_score']}"
        )