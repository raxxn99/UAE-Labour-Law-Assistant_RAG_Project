import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# CONFIGURATION (Edit these for D2 experiments)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHROMA_DB_PATH = 'chroma_db'          # Now actually used ✅

def load_documents(data_dir):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            print(f'Loading {filename}...')
            with open(filepath, 'r', encoding='utf-8') as f:
                documents.append({'content': f.read(), 'source': filename})
    print(f'Loaded {len(documents)} documents')
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '.', ' ', '']
    )
    chunks = []
    for doc in documents:
        for i, chunk in enumerate(splitter.split_text(doc['content'])):
            chunks.append({
                'text': chunk,
                'source': doc['source'],
                'chunk_id': f"{doc['source']}_chunk_{i}"
            })
    print(f'Created {len(chunks)} chunks')
    return chunks

def create_embeddings_and_store(chunks):
    print(f'Loading embedding model: {EMBEDDING_MODEL}...')
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ✅ Fix 1: PersistentClient so data survives between runs
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection('uae_labour_law')

    texts = [c['text'] for c in chunks]
    ids   = [c['chunk_id'] for c in chunks]
    metas = [{'source': c['source'], 'chunk_index': str(i)} for i, c in enumerate(chunks)]

    # ✅ Fix 2: Batch encode (much faster)
    print('Creating embeddings...')
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # ✅ Fix 3: upsert instead of add (safe to re-run)
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
    print(f'Stored {len(chunks)} chunks in ChromaDB at: {CHROMA_DB_PATH}/')

def main():
    print('=== Starting Data Ingestion Pipeline ===')
    
    # ✅ Fix 4: Correct folder name matching your actual structure
    documents = load_documents('data/processed_data')

    
    if not documents:
        print('❌ No .txt files found. Has Salma run data_cleaning.py yet?')
        return

    chunks = chunk_documents(documents)
    create_embeddings_and_store(chunks)
    print('=== Ingestion Complete ===')

if __name__ == '__main__':
    main()