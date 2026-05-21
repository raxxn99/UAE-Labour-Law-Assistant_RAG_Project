import chromadb

client = chromadb.PersistentClient(path='chroma_db')
collection = client.get_collection('uae_labour_law')

print(f'Total chunks stored: {collection.count()}')