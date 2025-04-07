# db_operations.py
import chromadb

chroma_path = "./chromadb"
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_collection("email_texts")

def get_all_data():
    """Returns all data in the ChromaDB collection."""
    return collection.get(include=["documents"])

# Example usage (in a separate main.py or similar):
all_data = get_all_data()
print(all_data)