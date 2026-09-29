from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.llm import get_llm


def build_rag_agent(persist_dir: str | None = None, llm=None):
    persist_dir = persist_dir or settings.chroma_persist_dir

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
    )
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=settings.chroma_collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.rag_top_k})

    llm = llm or get_llm()

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