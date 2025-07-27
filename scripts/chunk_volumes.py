import os
import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm 

RAW_FOLDER = Path("data/raw_volumes/ExportedVolumes")
CHUNK_FOLDER = Path("data/processed_chunks")

def chunk_volumes():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # character count, not words
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    for file in tqdm(RAW_FOLDER.glob("*.txt"), desc="Chunking Volumes"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs = loader.load()
        
        split_docs = splitter.split_documents(docs)
        for i, doc in enumerate(split_docs):
            doc.metadata["volume"] = file.stem
            doc.metadata["chunk_id"] = f"{file.stem}_chunk{i+1}"
            
        output_file = CHUNK_FOLDER / f"{file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([{
                "text": doc.page_content,
                "metadata": doc.metadata
            } for doc in split_docs], f, indent=2)
        
if __name__ == "__main__":
    chunk_volumes()            
        
