"""
AI Document Assistant - Streamlit Frontend
Interactive UI for document upload, management, and Q&A.
"""

import os
import requests
import streamlit as st
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background: #0f1117; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
    }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #e8e8f0 !important;
    }

    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #e8e8f0;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1rem;
        color: #6b7280;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    .accent { color: #7c6af7; }

    .answer-box {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-left: 3px solid #7c6af7;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d1d5db;
    }

    .citation-card {
        background: #161928;
        border: 1px solid #2a2d42;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }

    .citation-header {
        font-family: 'IBM Plex Mono', monospace;
        color: #7c6af7;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .citation-text {
        color: #9ca3af;
        font-style: italic;
        line-height: 1.5;
    }

    .score-badge {
        display: inline-block;
        background: #2d1f6b;
        color: #a78bfa;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        margin-left: 8px;
    }

    .doc-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
    }

    .doc-name {
        font-family: 'IBM Plex Mono', monospace;
        color: #e8e8f0;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .doc-meta {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .status-ok { color: #34d399; }
    .status-err { color: #f87171; }

    .stButton > button {
        background: #7c6af7 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background: #6b59e6 !important;
        transform: translateY(-1px) !important;
    }

    .stTextArea textarea, .stTextInput input {
        background: #1e2130 !important;
        border: 1px solid #2d3148 !important;
        color: #e8e8f0 !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .stSelectbox > div > div {
        background: #1e2130 !important;
        border: 1px solid #2d3148 !important;
        color: #e8e8f0 !important;
    }

    .divider {
        border: none;
        border-top: 1px solid #2d3148;
        margin: 1.5rem 0;
    }

    .thinking-msg {
        font-family: 'IBM Plex Mono', monospace;
        color: #7c6af7;
        font-size: 0.85rem;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


# ── API Helpers ───────────────────────────────────────────────────────────────

def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_upload(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def api_list_documents() -> dict:
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    r.raise_for_status()
    return r.json()


def api_delete_document(doc_id: str) -> dict:
    r = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_query(question: str, doc_id: Optional[str], top_k: int, model: str) -> dict:
    payload = {
        "question": question,
        "doc_id": doc_id if doc_id != "All Documents" else None,
        "top_k": top_k,
        "model": model,
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Session State ─────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []


def refresh_documents():
    try:
        data = api_list_documents()
        st.session_state.documents = data.get("documents", [])
    except Exception:
        st.session_state.documents = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title">📄 DocAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">RAG Document Assistant</div>', unsafe_allow_html=True)

    # Health
    health = api_health()
    if health:
        st.markdown(f'<span class="status-ok">● API Online</span> &nbsp; '
                    f'<span style="color:#6b7280;font-size:0.8rem;">{health["total_documents"]} docs · '
                    f'{health["total_chunks"]} chunks</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">● API Offline</span>', unsafe_allow_html=True)
        st.warning("Make sure the backend is running:\n```\npython main.py\n```")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Upload ──────────────────────────────────────────────────────────────
    st.markdown("**Upload Document**")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded and st.button("📤 Index Document"):
        with st.spinner("Processing and indexing..."):
            try:
                result = api_upload(uploaded.read(), uploaded.name)
                st.success(f"✅ Indexed **{result['chunks_indexed']}** chunks")
                refresh_documents()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Documents ───────────────────────────────────────────────────────────
    st.markdown("**Indexed Documents**")

    if st.button("🔄 Refresh", use_container_width=True):
        refresh_documents()

    if not st.session_state.documents:
        refresh_documents()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.expander(f"📁 {doc['filename'][:28]}...' if len(doc['filename']) > 28 else doc['filename']"):
                st.markdown(
                    f'<div class="doc-meta">'
                    f'{doc["total_chunks"]} chunks · {doc["total_pages"]} pages<br>'
                    f'<code style="font-size:0.7rem;color:#4b5563">{doc["doc_id"][:16]}...</code>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button("🗑️ Delete", key=f"del_{doc['doc_id']}"):
                    try:
                        api_delete_document(doc["doc_id"])
                        st.success("Removed")
                        refresh_documents()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
    else:
        st.markdown('<div style="color:#4b5563;font-size:0.85rem;">No documents indexed yet.</div>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Settings ─────────────────────────────────────────────────────────────
    st.markdown("**Settings**")

    model_choice = st.selectbox(
        "LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo"],
        index=0,
    )

    top_k = st.slider("Chunks to Retrieve", min_value=1, max_value=15, value=5)

    doc_names = ["All Documents"] + [d["filename"] for d in st.session_state.documents]
    doc_ids_map = {d["filename"]: d["doc_id"] for d in st.session_state.documents}
    selected_doc_name = st.selectbox("Search Scope", doc_names)
    selected_doc_id = doc_ids_map.get(selected_doc_name)


# ── Main Area ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero-title">AI Document Assistant</div>'
    '<div class="hero-subtitle">Ask anything about your uploaded documents</div>',
    unsafe_allow_html=True
)

# Chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)
        if entry.get("citations"):
            with st.expander(f"📚 {len(entry['citations'])} Sources"):
                for c in entry["citations"]:
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<div class="citation-header">'
                        f'📄 {c["filename"]} — Page {c["page_num"]}'
                        f'<span class="score-badge">{c["relevance_score"]:.3f}</span>'
                        f'</div>'
                        f'<div class="citation-text">"{c["excerpt"]}"</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# Query input
question = st.chat_input("Ask a question about your documents...")

if question:
    if not health:
        st.error("API is offline. Please start the backend server.")
    elif not st.session_state.documents:
        st.warning("No documents indexed. Upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = api_query(
                        question=question,
                        doc_id=selected_doc_id,
                        top_k=top_k,
                        model=model_choice,
                    )

                    answer = result["answer"]
                    citations = result.get("citations", [])

                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    if citations:
                        with st.expander(f"📚 {len(citations)} Sources (retrieved {result['retrieved_chunks']} chunks)"):
                            for c in citations:
                                st.markdown(
                                    f'<div class="citation-card">'
                                    f'<div class="citation-header">'
                                    f'📄 {c["filename"]} — Page {c["page_num"]}'
                                    f'<span class="score-badge">{c["relevance_score"]:.3f}</span>'
                                    f'</div>'
                                    f'<div class="citation-text">"{c["excerpt"]}"</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "citations": citations,
                    })

                except requests.HTTPError as e:
                    detail = e.response.json().get("detail", str(e))
                    st.error(f"Query failed: {detail}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
