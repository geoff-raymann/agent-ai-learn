import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_utils import ingest_pdf, answer_with_rag, list_documents

DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

app = FastAPI(title="Day 3 Multi-Document RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Day 3 Multi-Document RAG API is running",
        "endpoints": ["/upload", "/query", "/documents"]
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for now."
        )

    save_path = os.path.join(DOCS_DIR, file.filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        result = ingest_pdf(save_path)
        return {
            "message": "Upload and ingestion successful",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_document(payload: dict):
    question = payload.get("question")
    top_k = payload.get("top_k")
    max_context_chunks = payload.get("max_context_chunks", 3)
    doc_name = payload.get("doc_name")

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    try:
        result = answer_with_rag(
            query=question,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
            doc_name=doc_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    return list_documents()