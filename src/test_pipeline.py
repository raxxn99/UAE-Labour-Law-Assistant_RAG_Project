import os
import sys

# ── Fix all paths ──────────────────────────────────────────────
# Get the project root folder (wherever this file is saved)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

# Fix ChromaDB path — must be absolute so it always finds the db
os.chdir(ROOT)

# ── Load .env file ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

# ── Check API key exists ───────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    print("   Create a .env file in the project root with:")
    print("   GEMINI_API_KEY=your-key-here")
    sys.exit(1)
else:
    print("✅ GEMINI_API_KEY found")

# ── Check ChromaDB exists ──────────────────────────────────────
import chromadb
try:
    client     = chromadb.PersistentClient(path='chroma_db')
    collection = client.get_collection('uae_labour_law')
    print(f"✅ ChromaDB loaded — {collection.count()} chunks available")
except Exception as e:
    print(f"❌ ChromaDB error: {e}")
    sys.exit(1)

# ── Import pipeline ────────────────────────────────────────────
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

# ── Run test questions ─────────────────────────────────────────
questions = [
    "What is the minimum annual leave in the UAE?",
    "How is end of service gratuity calculated?",
    "What are the rules for probation period?",
    "What is the maximum working hours per week?",
    "How can an employee file a labour complaint?"
]

print("\n" + "="*60)
print("  RUNNING FULL PIPELINE TEST")
print("="*60)

for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    print("-"*60)
    try:
        result = answer_question(q)
        print(result['answer'])
    except Exception as e:
        print(f"❌ Error on Q{i}: {e}")
    print("-"*60)

print("\n✅ Test complete")