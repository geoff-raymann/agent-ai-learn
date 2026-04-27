import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

import fitz  # pymupdf
import numpy as np
import pyarrow as pa
import lancedb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_PATH = "lancedb_data"
TABLE_NAME = "rag_chunks"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

db = lancedb.connect(DB_PATH)


def get_or_create_table():
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("doc_name", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("page_numbers", pa.string()),
        pa.field("title", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
    ])

    try:
        return db.open_table(TABLE_NAME)
    except Exception:
        return db.create_table(TABLE_NAME, schema=schema)


table = get_or_create_table()


def embed_texts(texts: List[str]) -> List[List[float]]:
    return embedding_model.encode(texts).astype(np.float32).tolist()


def parse_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Extract text with pymupdf and split into overlapping chunks.
    No ML models — fast and memory-safe.
    """
    filename = Path(pdf_path).name
    records = []

    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue

        # Sliding window chunking
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                records.append({
                    "id": str(uuid.uuid4()),
                    "doc_name": filename,
                    "chunk_text": chunk,
                    "page_numbers": str(page_num),
                    "title": "",
                })
            start += chunk_size - overlap
    doc.close()
    return records


def reset_doc_chunks(doc_name: str):
    global table
    try:
        table.delete(f"doc_name = '{doc_name}'")
    except Exception:
        pass


def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    global table

    filename = Path(pdf_path).name
    reset_doc_chunks(filename)

    records = parse_and_chunk_pdf(pdf_path)
    if not records:
        raise ValueError("No chunks were produced from the uploaded PDF.")

    vectors = embed_texts([r["chunk_text"] for r in records])

    for r, v in zip(records, vectors):
        r["vector"] = v

    table.add(records)

    return {
        "file": filename,
        "chunks_ingested": len(records)
    }


def retrieve_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    query_vector = embed_texts([query])[0]

    results = (
        table.search(query_vector)
        .limit(top_k)
        .to_list()
    )

    return results


def build_context(chunks: List[Dict[str, Any]], max_chars_per_chunk: int = 200) -> str:
    blocks = []
    for c in chunks:
        src = f'{c["doc_name"]} pages {c.get("page_numbers", "[]")}'
        title = c.get("title", "")
        text = c["chunk_text"][:max_chars_per_chunk]
        blocks.append(
            f"[Source: {src}; title: {title}]\n{text}"
        )
    return "\n\n".join(blocks)


def answer_with_rag(query: str, top_k: int = 3) -> Dict[str, Any]:
    retrieved = retrieve_chunks(query, top_k=top_k)
    context = build_context(retrieved)

    prompt = f"""
You are a precise document assistant.

Use ONLY the supplied context.
If the answer is not in the context, say exactly:
"I could not find that in the uploaded document."

Context:
{context}

Question:
{query}

Return JSON with this shape:
{{
  "answer": "short answer",
  "sources": ["source 1", "source 2"]
}}
"""

    response = ollama_client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You answer only from retrieved document context and return valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)

    return {
        "answer": parsed.get("answer", ""),
        "sources": parsed.get("sources", []),
        "retrieved_chunks": retrieved
    }