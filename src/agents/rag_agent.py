"""RAG agent — grounds policy answers in the Chroma vector store."""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.llm import get_llm


def _fresh_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
    )


def build_rag_agent(persist_dir: str | None = None, llm=None):
    persist_dir = persist_dir or settings.chroma_persist_dir
    llm = llm or get_llm()

    # Disable Chroma's in-memory cache so every call reads from disk.
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=_fresh_embeddings(),
        collection_name=settings.chroma_collection_name,
        client_settings=ChromaSettings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        ),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.rag_top_k})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the following context from company policy documents.
    If the context does not contain the answer, say "I cannot find this information in the provided documents."
    Always cite the relevant section.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain