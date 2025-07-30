import os
import re
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st

st.title("Story Copilot")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

persist_directory = "chroma_db"
collection_name = "story_docs"
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=persist_directory,
)

def extract_volume_number(volume_str):
    match = re.search(r'season\s+(\d+)\s+volume\s+(\d+)', volume_str.lower())
    if match:
        season = int(match.group(1))
        volume = int(match.group(2))
        return season * 1000 + volume
    return float('inf')

def get_metadata_filter_and_clean_query(query):
    season_volume_match = re.search(r'season\s+\d+\s+volume\s+\d+', query, re.IGNORECASE)
    if season_volume_match:
        full_season_volume = season_volume_match.group(0).lower()
        cleaned_query = re.sub(re.escape(full_season_volume), '', query, flags=re.IGNORECASE).strip()
        return {"volume": full_season_volume}, cleaned_query
    else:
        season_match = re.search(r'season\s+(\d+)', query, re.IGNORECASE)
        volume_match = re.search(r'volume\s+(\d+)', query, re.IGNORECASE)
        season = season_match.group(1) if season_match else None
        volume = volume_match.group(1) if volume_match else None
        cleaned_query = re.sub(r'season\s+\d+', '', query, flags=re.IGNORECASE)
        cleaned_query = re.sub(r'volume\s+\d+', '', cleaned_query, flags=re.IGNORECASE).strip()
        if season and volume:
            return {"volume": f"season {season} volume {volume}"}, cleaned_query
        elif season:
            # For only season
            return lambda metadata: f"season {season}" in metadata.get("volume", "").lower(), cleaned_query
        else:
            return None, query.strip()

query = st.text_input("Ask me something")

if st.button("Send") and query.strip():
    metadata_filter, cleaned_query = get_metadata_filter_and_clean_query(query.strip())

    if callable(metadata_filter):
        all_docs = vectorstore.similarity_search(cleaned_query, k=100)
        docs = [doc for doc in all_docs if metadata_filter(doc.metadata)]
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"filter": metadata_filter, "k": 20})
        docs = retriever.get_relevant_documents(cleaned_query)

    docs.sort(key=lambda doc: extract_volume_number(doc.metadata.get("volume", "")))

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an assistant that answers questions based on the following context from story volumes. Don't say based on the context. Answer as if you are a knowledge base of the stories
Context:
{context}

Question:
{cleaned_query}

Answer:
"""

    st.write("Running Story Copilot...")
    response = model.generate_content(prompt)
    answer = response.text
    st.write("**Answer:**")
    st.write(answer)

    st.write("**Source Documents:**")
    for doc in docs:
        vol = doc.metadata.get("volume", "N/A")
        st.write(f"- Volume: {vol}")

