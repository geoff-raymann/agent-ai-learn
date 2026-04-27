import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_docling_lancedb import ingest_pdf, answer_with_rag

logging.basicConfig(level=logging.INFO)

DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

app = FastAPI(title="Day 2 RAG with Docling + LanceDB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Docling + LanceDB RAG API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for now.")

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
        logging.exception("Upload failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_document(payload: dict):
    question = payload.get("question")
    top_k = payload.get("top_k", 5)

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    try:
        return answer_with_rag(question, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))