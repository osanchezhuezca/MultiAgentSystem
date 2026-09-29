# src/ingestion/vector_store.py
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vector_store(chunks, persist_dir: str = "data/processed/chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="policy_docs"
    )
    return vectorstore