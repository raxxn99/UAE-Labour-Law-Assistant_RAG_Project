import os
import sys
from pathlib import Path
import time
# =========================================================
# FIND PROJECT ROOT SAFELY
# =========================================================

CURRENT_FILE_DIR = Path(__file__).resolve().parent

# Case 1: test_pipeline.py is in project root
if (CURRENT_FILE_DIR / "src").exists():
    PROJECT_ROOT = CURRENT_FILE_DIR

# Case 2: test_pipeline.py is inside src/
elif (CURRENT_FILE_DIR.parent / "src").exists():
    PROJECT_ROOT = CURRENT_FILE_DIR.parent

else:
    raise RuntimeError(
        "Could not find project root. Make sure the project has a src/ folder."
    )

SRC_DIR = PROJECT_ROOT / "src"

# Add src/ to Python path
sys.path.insert(0, str(SRC_DIR))

# Change working directory to project root
os.chdir(PROJECT_ROOT)

print(f"✅ Project root: {PROJECT_ROOT}")
print(f"✅ Source folder: {SRC_DIR}")


# =========================================================
# LOAD .ENV FILE
# =========================================================

from dotenv import load_dotenv

ENV_PATH = PROJECT_ROOT / ".env"

if not ENV_PATH.exists():
    print("❌ .env file not found")
    print(f"Expected location: {ENV_PATH}")
    print("Create .env in the project root with:")
    print("GEMINI_API_KEY=your-key-here")
    sys.exit(1)

load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    print(f"Checked file: {ENV_PATH}")
    print("Your .env should contain:")
    print("GEMINI_API_KEY=your-key-here")
    sys.exit(1)

if api_key.startswith("="):
    print("❌ GEMINI_API_KEY starts with an extra '='")
    print("Fix .env from:")
    print("GEMINI_API_KEY==AIza...")
    print("to:")
    print("GEMINI_API_KEY=AIza...")
    sys.exit(1)

print("✅ GEMINI_API_KEY found")
print(f"Key starts with: {api_key[:6]}")
print(f"Key length: {len(api_key)}")


# =========================================================
# CHECK CHROMADB
# =========================================================

import chromadb

try:
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("uae_labour_law")
    print(f"✅ ChromaDB loaded — {collection.count()} chunks available")
except Exception as e:
    print(f"❌ ChromaDB error: {e}")
    print("Make sure Rana has run:")
    print("python src/ingest.py")
    sys.exit(1)


# =========================================================
# IMPORT PIPELINE
# =========================================================

try:
    from retrieval import retrieve
    print("✅ retrieval.py imported")
except Exception as e:
    print(f"❌ retrieval.py import failed: {e}")
    sys.exit(1)

try:
    from generate import answer_question
    print("✅ generate.py imported")
except Exception as e:
    print(f"❌ generate.py import failed: {e}")
    sys.exit(1)


# =========================================================
# RUN TEST QUESTIONS
# =========================================================

questions = [
    "What is the minimum annual leave in the UAE?",
    "How is end of service gratuity calculated?",
    "What are the rules for probation period?",
    "What is the maximum working hours per week?",
    "How can an employee file a labour complaint?"
]

print("\n" + "=" * 60)
print("  RUNNING FULL PIPELINE TEST")
print("=" * 60)

for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    print("-" * 60)

    try:
        result = answer_question(q)
        print(result["answer"])
    except Exception as e:
        print(f"❌ Error on Q{i}: {e}")

    print("-" * 60)
    time.sleep(10)  # Sleep between questions to avoid hitting API rate limits

print("\n✅ Test complete")