import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Document Copilot", page_icon="📚", layout="wide")

# -------- helpers --------
def api_get(path):
    r = requests.get(f"{API_BASE_URL}{path}", timeout=15)
    r.raise_for_status()
    return r.json()

def api_post(path, payload=None, files=None):
    r = requests.post(f"{API_BASE_URL}{path}", json=payload, files=files, timeout=90)
    r.raise_for_status()
    return r.json()

def get_documents():
    try:
        return api_get("/documents")
    except Exception as e:
        return {"documents": [], "error": str(e)}

def get_memory():
    try:
        return api_get("/memory")
    except Exception:
        return {"memory": []}

def ask(question, doc_name=None, top_k=None, max_context_chunks=3):
    payload = {"question": question, "max_context_chunks": max_context_chunks}
    if doc_name and doc_name != "All documents":
        payload["doc_name"] = doc_name
    if top_k:
        payload["top_k"] = top_k
    return api_post("/query", payload=payload)

def upload_pdf(file):
    return api_post("/upload", files={"file": (file.name, file.getvalue(), "application/pdf")})

# -------- state --------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# -------- sidebar --------
with st.sidebar:
    st.title("⚙️ Controls")
    docs = get_documents().get("documents", [])
    options = ["All documents"] + [d["doc_name"] for d in docs]
    selected_doc = st.selectbox("Search scope", options)

    max_context_chunks = st.slider("Max context chunks", 1, 6, 3)
    use_manual = st.checkbox("Manual top_k")
    top_k = st.slider("Top K", 1, 15, 6) if use_manual else None

    st.divider()
    st.subheader("📤 Upload PDF")
    f = st.file_uploader("Upload", type=["pdf"], label_visibility="collapsed")
    if f and st.button("Upload and index", use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                res = upload_pdf(f)
                st.success("Indexed")
                st.json(res)
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("🧠 Memory")
    st.caption("Assistant remembers recent questions.")
    mem = get_memory().get("memory", [])
    if mem:
        for item in mem[-5:][::-1]:
            st.markdown(f"**Q:** {item.get('question','')}")
            st.caption(item.get("answer","")[:120] + ("..." if len(item.get("answer",""))>120 else ""))
    else:
        st.caption("No conversation memory yet.")

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

# -------- main --------
st.title("📚 RAG Document Copilot")
st.caption("Ask grounded questions across PDFs with citations, memory, and hybrid retrieval.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("Sources")
            for s in msg["sources"]:
                st.markdown(f"- `{s}`")

q = st.chat_input("Ask your documents...")

if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(q, selected_doc, top_k, max_context_chunks)
                ans = result.get("answer","")
                st.markdown(ans)
                if result.get("memory_used"):
                    st.info("Used recent conversation memory to understand your question.")
                sources = result.get("sources",[])
                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.markdown(f"- `{s}`")
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":ans,
                    "sources":sources
                })
                st.session_state.last_result = result
            except Exception as e:
                st.error(str(e))

if st.session_state.last_result:
    r = st.session_state.last_result
    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    m = r.get("metrics",{})
    c1.metric("Top K", m.get("top_k","-"))
    c2.metric("Retrieved", m.get("retrieved_count","-"))
    c3.metric("Used", m.get("selected_count","-"))
    c4.metric("Latency", f"{m.get('response_seconds','-')}s")

    with st.expander("Debug JSON"):
        st.json(r)