import os
import ssl
import uuid
import re
import json
from typing import List, Dict, Any

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Corporate network / HuggingFace setup
# ─────────────────────────────────────────────────────────────

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file.")

groq_client = Groq(api_key=groq_api_key)


# ─────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and compress whitespace for matching/reranking."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def clean_title(title: str) -> str:
    """
    Remove emoji/symbol noise from titles for stronger matching.
    Example: 'Conditional Edges🎏' -> 'conditional edges'
    """
    title = normalize_text(title)
    title = re.sub(r"[^a-z0-9\s\-_/]", "", title)
    return normalize_text(title)


def extract_title(text: str) -> str:
    """
    Use the first non-empty short line as the page title.
    Works well for slide decks.
    """
    for line in text.splitlines():
        clean = line.strip()
        if clean and len(clean) <= 80:
            return clean
    return "Untitled"


def rewrite_query(query: str) -> str:
    """
    Normalize common definition-style questions into retrieval-friendly text.
    """
    q = normalize_text(query)

    replacements = {
        "what is ": "",
        "what are ": "",
        "define ": "",
        "explain ": "",
        "describe ": "",
        "meaning of ": "",
        "?": "",
    }

    for old, new in replacements.items():
        q = q.replace(old, new)

    q = q.strip()

    # Helpful expansions for LangGraph-style concepts
    if q in {"langgraph state", "state"}:
        return "state in langgraph definition"

    if q in {"stategraph", "langgraph stategraph"}:
        return "stategraph langgraph class build compile graph"

    if q in {"conditional edges", "langgraph conditional edges"}:
        return "conditional edges decide next node conditions current state"

    if q in {"toolnode", "tool node"}:
        return "toolnode special node run tool output state"

    return q


# ─────────────────────────────────────────────────────────────
# PDF extraction + chunking
# ─────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            cleaned = text.strip()
            pages.append({
                "page_number": i + 1,
                "text": cleaned,
                "title": extract_title(cleaned)
            })

    return pages


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def is_slide_like_document(pages: List[Dict[str, Any]]) -> bool:
    """
    If most pages are short, treat each page as one chunk.
    Great for slide decks like LangGraph slides.
    """
    if not pages:
        return False

    short_pages = sum(1 for p in pages if len(p["text"]) < 1200)
    return (short_pages / len(pages)) >= 0.7


# ─────────────────────────────────────────────────────────────
# Chroma ingestion
# ─────────────────────────────────────────────────────────────

def clear_collection_for_file(filename: str) -> None:
    """
    Prevent duplicate ingestion of the same file.
    """
    try:
        existing = collection.get(where={"source": filename})
        ids = existing.get("ids", [])

        if ids:
            collection.delete(ids=ids)

    except Exception:
        pass


def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    filename = os.path.basename(pdf_path)
    pages = extract_pdf_text(pdf_path)

    if not pages:
        raise ValueError("No extractable text found in PDF.")

    clear_collection_for_file(filename)

    docs = []
    metadatas = []
    ids = []

    slide_mode = is_slide_like_document(pages)

    for page in pages:
        if slide_mode:
            page_chunks = [page["text"]]
        else:
            page_chunks = chunk_text(page["text"])

        for idx, chunk in enumerate(page_chunks):
            title = page["title"]

            docs.append(chunk)
            metadatas.append({
                "source": filename,
                "page": page["page_number"],
                "chunk_index": idx,
                "title": title,
                "title_clean": clean_title(title),
                "doc_type": "slides" if slide_mode else "document"
            })
            ids.append(str(uuid.uuid4()))

    embeddings = embedding_model.encode(docs).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {
        "file": filename,
        "pages_ingested": len(pages),
        "chunks_ingested": len(docs),
        "mode": "page-level" if slide_mode else "character-chunked"
    }


# ─────────────────────────────────────────────────────────────
# Retrieval + reranking
# ─────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    rewritten_query = rewrite_query(query)
    query_embedding = embedding_model.encode([rewritten_query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(docs, metas, distances):
        retrieved.append({
            "text": doc,
            "metadata": meta,
            "distance": float(distance)
        })

    return retrieved


def simple_rerank(query: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hybrid reranker:
    - title match
    - phrase match
    - lexical overlap
    - embedding distance bonus
    """
    q_rewritten = normalize_text(rewrite_query(query))
    q_terms = set(q_rewritten.split())

    for item in retrieved:
        text = normalize_text(item["text"])
        title_raw = item["metadata"].get("title", "")
        title = clean_title(title_raw)
        distance = item["distance"]

        score = 0.0

        # Exact title match
        if q_rewritten == title:
            score += 10.0

        # Phrase inside title/text
        if q_rewritten in title:
            score += 8.0

        if q_rewritten in text:
            score += 5.0

        # Concept-specific boosts
        if "state" in q_terms and title == "state":
            score += 8.0

        if "stategraph" in q_terms and title == "stategraph":
            score += 8.0

        if "conditional" in q_terms and "edges" in q_terms and title == "conditional edges":
            score += 8.0

        if "toolnode" in q_terms and title == "toolnode":
            score += 8.0

        # Lexical overlap
        title_terms = set(title.split())
        text_terms = set(text.split())

        score += len(q_terms.intersection(title_terms)) * 2.5
        score += len(q_terms.intersection(text_terms)) * 0.5

        # Lower vector distance is better
        score += max(0.0, 2.0 - distance)

        item["rerank_score"] = round(score, 4)

    return sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)


def select_context_chunks(
    reranked_chunks: List[Dict[str, Any]],
    query: str,
    max_context_chunks: int = 3
) -> List[Dict[str, Any]]:
    """
    Select only chunks that should be sent to the LLM.

    Definition questions usually need only the strongest chunk.
    Comparison questions need multiple chunks.
    """
    if not reranked_chunks:
        return []

    q = normalize_text(query)

    definition_triggers = [
        "what is",
        "what are",
        "define",
        "explain",
        "meaning of",
        "describe"
    ]

    comparison_triggers = [
        "difference",
        "compare",
        "versus",
        "vs",
        "distinguish"
    ]

    if any(trigger in q for trigger in comparison_triggers):
        selected = [reranked_chunks[0]]
        top_score = reranked_chunks[0].get("rerank_score", 0)

        for chunk in reranked_chunks[1:max_context_chunks]:
            chunk_score = chunk.get("rerank_score", 0)

            # Include only strongly relevant comparison chunks
            if top_score > 0 and chunk_score >= top_score * 0.60:
                selected.append(chunk)

        return selected

    if any(trigger in q for trigger in definition_triggers):
        return [reranked_chunks[0]]

    selected = [reranked_chunks[0]]
    top_score = reranked_chunks[0].get("rerank_score", 0)

    for chunk in reranked_chunks[1:max_context_chunks]:
        chunk_score = chunk.get("rerank_score", 0)

        if top_score > 0 and chunk_score >= top_score * 0.75:
            selected.append(chunk)

    return selected


def build_context(selected_chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for item in selected_chunks:
        meta = item["metadata"]

        source_label = (
            f'{meta["source"]} page {meta["page"]} '
            f'(title: {meta.get("title", "Untitled")})'
        )

        context_blocks.append(
            f"[Source: {source_label}]\n{item['text']}"
        )

    return "\n\n".join(context_blocks)


def build_source_list(selected_chunks: List[Dict[str, Any]]) -> List[str]:
    sources = []

    for item in selected_chunks:
        meta = item["metadata"]

        sources.append(
            f'{meta["source"]} page {meta["page"]} '
            f'(title: {meta.get("title", "Untitled")})'
        )

    return list(dict.fromkeys(sources))


# ─────────────────────────────────────────────────────────────
# RAG answer generation
# ─────────────────────────────────────────────────────────────

def answer_with_rag(
    query: str,
    top_k: int = 6,
    max_context_chunks: int = 3
) -> Dict[str, Any]:
    retrieved = retrieve_chunks(query, top_k=top_k)
    reranked = simple_rerank(query, retrieved)

    selected_chunks = select_context_chunks(
        reranked_chunks=reranked,
        query=query,
        max_context_chunks=max_context_chunks
    )

    context = build_context(selected_chunks)
    selected_sources = build_source_list(selected_chunks)

    if not selected_chunks:
        return {
            "answer": "I could not find that in the uploaded document.",
            "sources": [],
            "retrieved_chunks": retrieved,
            "reranked_chunks": [],
            "selected_context_chunks": []
        }

    prompt = f"""
You are a precise document assistant.

Use ONLY the supplied context.
If the answer is not present in the context, say exactly:
"I could not find that in the uploaded document."

Rules:
- Prefer the most directly relevant source.
- Be concise and accurate.
- Do not mention sources that were not used.
- If two concepts are similar, distinguish them clearly.
- Only cite sources included in the supplied context.
- Return valid JSON only.

Context:
{context}

Question:
{query}

Return JSON with this shape:
{{
  "answer": "short answer here",
  "sources": ["source 1", "source 2"]
}}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer only from retrieved document context "
                    "and return valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "answer": raw,
            "sources": selected_sources
        }

    answer = parsed.get("answer", "")
    sources = parsed.get("sources", selected_sources)

    # Safety: never allow model to cite sources outside selected context
    valid_sources = set(selected_sources)
    sources = [s for s in sources if s in valid_sources]

    if not sources:
        sources = selected_sources

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
        "reranked_chunks": reranked[:max_context_chunks],
        "selected_context_chunks": selected_chunks
    }