import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_utils import ingest_pdf, answer_with_rag, list_documents

DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

app = FastAPI(title="Day 3 Memory RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory conversation store
conversation_memory = []


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_recent_context(limit=3):
    recent = conversation_memory[-limit:]
    lines = []

    for item in recent:
        lines.append(f'Q: {item["question"]}')
        lines.append(f'A: {item["answer"]}')

    return "\n".join(lines)


def enrich_query(question: str):
    """
    Lightweight conversational memory.
    If user uses pronouns like it/that/they,
    prepend recent conversation context.
    """
    q = question.lower()

    memory_words = ["it", "that", "they", "those", "them", "this"]

    if any(word in q.split() for word in memory_words):
        context = get_recent_context()

        if context:
            return f"""
Conversation context:
{context}

Current user question:
{question}
"""

    return question


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Day 3 Memory RAG API running",
        "endpoints": ["/upload", "/query", "/documents", "/memory"]
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF supported.")

    save_path = os.path.join(DOCS_DIR, file.filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    result = ingest_pdf(save_path)

    return {
        "message": "Upload successful",
        "details": result
    }


@app.get("/documents")
def documents():
    return list_documents()


@app.get("/memory")
def memory():
    return {
        "memory": conversation_memory[-10:]
    }


@app.post("/query")
async def query(payload: dict):
    question = payload.get("question")
    top_k = payload.get("top_k")
    max_context_chunks = payload.get("max_context_chunks", 3)
    doc_name = payload.get("doc_name")

    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    enriched_question = enrich_query(question)

    result = answer_with_rag(
        query=enriched_question,
        top_k=top_k,
        max_context_chunks=max_context_chunks,
        doc_name=doc_name
    )

    conversation_memory.append({
        "question": question,
        "answer": result["answer"]
    })

    # keep last 10 only
    if len(conversation_memory) > 10:
        conversation_memory.pop(0)

    result["memory_used"] = enriched_question != question

    return result