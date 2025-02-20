import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="backend/data/chroma_db")

# Create a dummy collection to trigger model download
collection = chroma_client.get_or_create_collection(name="test_collection")

# Test query to force model download
try:
    collection.query(query_texts=["test"], n_results=1)
    print("Model download completed successfully!")
except Exception as e:
    print(f"Error triggering model download: {e}")
