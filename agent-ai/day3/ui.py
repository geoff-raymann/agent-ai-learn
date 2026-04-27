import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Document Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #064e3b 100%);
            color: white;
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #d1fae5;
            margin-bottom: 0;
        }

        .status-card {
            padding: 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            background-color: #ffffff;
            box-shadow: 0 1px 5px rgba(15, 23, 42, 0.06);
            min-height: 98px;
        }

        .status-label {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: .3rem;
        }

        .status-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.82rem;
        }

        .source-pill {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            border-radius: 999px;
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .answer-box {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            margin-top: 0.4rem;
        }

        .doc-card {
            padding: .75rem .85rem;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            margin-bottom: .55rem;
        }

        .doc-title {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: .2rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            background: #f1f5f9;
        }

        div[data-testid="stSidebar"] {
            background-color: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────

def get_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "documents": [],
            "document_count": 0,
            "error": str(e)
        }


def ask_rag(question, doc_name=None, top_k=None, max_context_chunks=3):
    payload = {
        "question": question,
        "max_context_chunks": max_context_chunks
    }

    if doc_name and doc_name != "All documents":
        payload["doc_name"] = doc_name

    if top_k:
        payload["top_k"] = top_k

    response = requests.post(
        f"{API_BASE_URL}/query",
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def upload_pdf(file):
    files = {
        "file": (file.name, file.getvalue(), "application/pdf")
    }

    response = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        timeout=120
    )
    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────

def source_pills(sources):
    if not sources:
        st.caption("No citations returned.")
        return

    html = ""
    for source in sources:
        html += f'<span class="source-pill">{source}</span>'

    st.markdown(html, unsafe_allow_html=True)


def compact_metrics(metrics):
    if not metrics:
        return

    top_k = metrics.get("top_k", "-")
    retrieved = metrics.get("retrieved_count", "-")
    selected = metrics.get("selected_count", "-")
    latency = metrics.get("response_seconds", "-")
    doc_filter = metrics.get("doc_filter", "all_documents")

    cols = st.columns(5)
    cols[0].metric("Top K", top_k)
    cols[1].metric("Retrieved", retrieved)
    cols[2].metric("Used", selected)
    cols[3].metric("Latency", f"{latency}s")
    cols[4].metric("Scope", "All docs" if doc_filter == "all_documents" else "1 doc")


def render_doc_cards(documents):
    if not documents:
        st.info("No indexed documents yet. Upload a PDF to start.")
        return

    for doc in documents:
        st.markdown(
            f"""
            <div class="doc-card">
                <div class="doc-title">📄 {doc.get("doc_name")}</div>
                <div class="small-muted">
                    {doc.get("page_count")} pages · {doc.get("chunks")} chunks · {doc.get("doc_type")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_selected_context(chunks):
    if not chunks:
        st.info("No context was sent to the model. This usually means the no-answer gate blocked weak retrieval.")
        return

    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Untitled")
        page = meta.get("page", "?")
        source = meta.get("source", "Unknown")
        score = chunk.get("rerank_score", "-")

        with st.expander(f"{idx}. {title} · page {page} · score {score}", expanded=idx == 1):
            st.caption(f"Source: {source}")
            st.write(chunk.get("text", ""))


def render_retrieval_debug(chunks):
    if not chunks:
        st.info("No retrieved chunks.")
        return

    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Untitled")
        page = meta.get("page", "?")
        source = meta.get("source", "Unknown")

        with st.expander(f"{idx}. {title} · page {page}"):
            score_cols = st.columns(4)
            score_cols[0].metric("Semantic", chunk.get("semantic_score", "-"))
            score_cols[1].metric("Keyword", chunk.get("keyword_score", "-"))
            score_cols[2].metric("Boost", chunk.get("concept_boost", "-"))
            score_cols[3].metric("Final", chunk.get("rerank_score", "-"))

            st.caption(f"Source: {source}")
            st.write(chunk.get("text", ""))


def example_prompts():
    return [
        "What is ToolNode?",
        "What is the difference between State and StateGraph?",
        "Summarize LangGraph core concepts",
        "What are Conditional Edges?",
        "Who founded LangGraph?"
    ]


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Controls")

    docs_payload = get_documents()
    documents = docs_payload.get("documents", [])
    doc_options = ["All documents"] + [doc["doc_name"] for doc in documents]

    selected_doc = st.selectbox(
        "Search scope",
        options=doc_options,
        index=0,
        help="Search all indexed documents or restrict retrieval to one document."
    )

    with st.expander("Retrieval settings", expanded=False):
        max_context_chunks = st.slider(
            "Max context chunks",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            help="Maximum chunks sent to the LLM after reranking."
        )

        manual_top_k = st.checkbox("Set top_k manually", value=False)

        top_k = None
        if manual_top_k:
            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=15,
                value=6,
                step=1,
                help="Number of chunks retrieved before reranking."
            )

    st.divider()

    st.subheader("📤 Upload document")
    uploaded_file = st.file_uploader(
        "PDF only",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button("Upload and index", use_container_width=True):
            with st.spinner("Indexing document..."):
                try:
                    result = upload_pdf(uploaded_file)
                    st.success("Indexed successfully")
                    st.json(result)
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()

    st.subheader("📚 Indexed documents")
    if docs_payload.get("error"):
        st.error(docs_payload["error"])
    else:
        render_doc_cards(documents)

    st.divider()

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📚 RAG Document Copilot</div>
        <p class="hero-subtitle">
            Ask grounded questions across your PDFs with citations, hybrid retrieval scores, and no-answer protection.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

documents_count = len(documents)
total_chunks = sum(doc.get("chunks", 0) for doc in documents)
total_pages = sum(doc.get("page_count", 0) for doc in documents)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Documents</div>
            <div class="status-value">{documents_count}</div>
            <div class="small-muted">indexed PDFs</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Pages</div>
            <div class="status-value">{total_pages}</div>
            <div class="small-muted">available for search</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Chunks</div>
            <div class="status-value">{total_chunks}</div>
            <div class="small-muted">stored in vector DB</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m4:
    scope_label = "All documents" if selected_doc == "All documents" else "Filtered"
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Search Scope</div>
            <div class="status-value" style="font-size: 1.05rem;">{scope_label}</div>
            <div class="small-muted">{selected_doc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────

left_col, right_col = st.columns([0.64, 0.36], gap="large")

with left_col:
    st.subheader("💬 Chat")

    if not st.session_state.messages:
        st.info("Ask a question about your indexed documents, or try one of the examples below.")

        prompt_cols = st.columns(2)
        prompts = example_prompts()

        for i, prompt in enumerate(prompts):
            with prompt_cols[i % 2]:
                if st.button(prompt, use_container_width=True):
                    st.session_state.selected_prompt = prompt
                    st.rerun()

    chat_container = st.container(height=430, border=True)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("sources"):
                    st.caption("Sources")
                    source_pills(message["sources"])

    default_question = st.session_state.selected_prompt or ""
    question = st.chat_input("Ask your documents...")

    if default_question and not question:
        question = default_question
        st.session_state.selected_prompt = None

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask_rag(
                    question=question,
                    doc_name=selected_doc,
                    top_k=top_k,
                    max_context_chunks=max_context_chunks
                )

                answer = result.get("answer", "")
                sources = result.get("sources", [])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                st.session_state.last_response = result
                st.rerun()

            except Exception as e:
                error_msg = f"Request failed: {e}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
                st.error(error_msg)


with right_col:
    st.subheader("📌 Latest answer details")

    result = st.session_state.last_response

    if not result:
        st.info("Answer details will appear here after your first question.")
    else:
        st.markdown("**Citations**")
        source_pills(result.get("sources", []))

        st.write("")
        compact_metrics(result.get("metrics", {}))

        if result.get("answer") == "I could not find that in the uploaded document.":
            st.warning("No-answer gate triggered. Retrieval was too weak to answer safely.")

        with st.expander("Selected context", expanded=True):
            render_selected_context(result.get("selected_context_chunks", []))

# ─────────────────────────────────────────────────────────────
# Debug area
# ─────────────────────────────────────────────────────────────

if st.session_state.last_response:
    st.divider()

    with st.expander("🔍 Retrieval debug console", expanded=False):
        tabs = st.tabs(["Reranked chunks", "Metrics", "Raw JSON"])

        with tabs[0]:
            render_retrieval_debug(st.session_state.last_response.get("reranked_chunks", []))

        with tabs[1]:
            st.json(st.session_state.last_response.get("metrics", {}))

        with tabs[2]:
            st.json(st.session_state.last_response)