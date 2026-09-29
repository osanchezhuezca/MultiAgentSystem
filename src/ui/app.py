"""Streamlit chat UI for the multi-agent customer support assistant."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# --- Make project root importable when Streamlit runs this as a script ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import settings
from src.ingestion.build import build as build_vector_store
from src.agents.supervisor import build_supervisor_graph

# ---------- Page config ----------
st.set_page_config(
    page_title=settings.streamlit_page_title,
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_supervisor_graph()


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Policy Documents")
    uploaded = st.file_uploader(
        "Upload a policy PDF",
        type=["pdf"],
        help="New PDFs are embedded into the vector store immediately.",
    )

    if uploaded is not None:
        if st.button("Ingest PDF", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name} ..."):
                tmp_dir = Path(tempfile.mkdtemp())
                tmp_path = tmp_dir / uploaded.name
                tmp_path.write_bytes(uploaded.getbuffer())
                try:
                    n = build_vector_store(
                        pdf_dir=tmp_dir,
                        persist_dir=Path(settings.chroma_persist_dir),
                    )
                    st.success(f"Added {n} chunks from {uploaded.name}.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("### Example queries")
    st.markdown(
        "- *What is the current refund policy?*\n"
        "- *How many open high-priority tickets are there?*\n"
        "- *Give me a customer overview and the refund policy.*"
    )

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Provider: `{settings.llm_provider}`")


# ---------- Header ----------
st.title("Multi-Agent Customer Support Assistant")
st.caption(
    "Ask about customer profiles, ticket history, or company policies. "
    "The supervisor routes your question to the right specialist agent."
)


# ---------- Render history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("route"):
            with st.expander("Routing details", expanded=False):
                st.markdown(f"**Route:** `{msg['route']}`")
                if msg.get("sql_result"):
                    st.markdown("**SQL agent output:**")
                    st.code(msg["sql_result"], language="text")
                if msg.get("rag_result"):
                    st.markdown("**RAG agent output:**")
                    st.code(msg["rag_result"], language="text")


# ---------- Chat input ----------
if prompt := st.chat_input("Ask about customers or policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing to specialist agents..."):
            initial_state = {
                "query": prompt,
                "route": "",
                "sql_query": "",
                "rag_query": "",
                "sql_result": "",
                "rag_result": "",
                "final_answer": "",
            }
            try:
                graph = get_graph()
                result = graph.invoke(initial_state)
                answer = result.get("final_answer") or "_(no answer produced)_"
                route = result.get("route", "?")
            except Exception as e:
                answer = f"**Error:** {e}"
                route = "error"
                result = {}

        st.markdown(answer)

        with st.expander("Routing details", expanded=False):
            st.markdown(f"**Route:** `{route}`")
            if result.get("sql_result"):
                st.markdown("**SQL agent output:**")
                st.code(result["sql_result"], language="text")
            if result.get("rag_result"):
                st.markdown("**RAG agent output:**")
                st.code(result["rag_result"], language="text")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "route": route,
        "sql_result": result.get("sql_result", ""),
        "rag_result": result.get("rag_result", ""),
    })
