from sentence_transformers import SentenceTransformer
import os
import json
import chromadb
from chromadb.config import Settings

# Load text chunks and metadata
chunks_dir = "data/processed_chunks"
documents = []
metadatas = []

for filename in os.listdir(chunks_dir):
    with open(os.path.join(chunks_dir, filename), "r") as f:
        data = json.load(f)
        # Assuming each JSON file is a list of chunks:
        for chunk in data:
            documents.append(chunk["text"])
            # Attach metadata you want (e.g., volume, filename)
            metadata = chunk.get("metadata", {})
            print(metadata.get("volume"))
            metadata["filename"] = filename  # add filename for traceability
            metadatas.append(metadata)
            

# Load model
model = SentenceTransformer("BAAI/bge-small-en")  # or another free model

# Embed documents
embeddings = model.encode(documents, show_progress_bar=True)

# Initialize Chroma DB (stores in ./chroma_db folder)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("story_docs")

# Add documents + embeddings + metadata
collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))]
)


query = "What did Nidarshan do with the time machine?"

# Embed query and search
query_embedding = model.encode([query])
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

for meta in results["metadatas"][0]:
    print(meta)

print("🔍 Top results:")
for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {doc} (volume: {metadata.get('volume')}, file: {metadata.get('filename')})")
