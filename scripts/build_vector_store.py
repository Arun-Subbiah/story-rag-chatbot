from sentence_transformers import SentenceTransformer
import os
import json
import chromadb
from chromadb.config import Settings

chunks_dir = "data/processed_chunks"
documents = []
metadatas = []

for filename in os.listdir(chunks_dir):
    with open(os.path.join(chunks_dir, filename), "r") as f:
        data = json.load(f)
        for chunk in data:
            documents.append(chunk["text"])
            metadata = chunk.get("metadata", {})
            print(metadata.get("volume"))
            metadata["filename"] = filename 
            metadatas.append(metadata)
            

model = SentenceTransformer("BAAI/bge-small-en")  

embeddings = model.encode(documents, show_progress_bar=True)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("story_docs")

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

