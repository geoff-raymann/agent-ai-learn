import os
import ssl
import uuid
import re
import json
import time
from typing import List, Dict, Any, Optional, Set

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

NO_ANSWER_SCORE_THRESHOLD = 4.0

# Hybrid scoring weights
SEMANTIC_WEIGHT = 1.0
KEYWORD_WEIGHT = 1.2
CONCEPT_BOOST_WEIGHT = 1.0

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
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def clean_for_matching(text: str) -> str:
    """
    Lowercase, remove most symbols/emojis, and compress whitespace.
    Useful for robust keyword matching.
    """
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\s\-_/]", " ", text)
    return normalize_text(text)


def clean_title(title: str) -> str:
    return clean_for_matching(title)


def extract_title(text: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean and len(clean) <= 80:
            return clean
    return "Untitled"


def get_terms(text: str) -> Set[str]:
    cleaned = clean_for_matching(text)
    terms = set(cleaned.split())

    # Tiny stopword list to reduce noisy keyword overlap.
    stopwords = {
        "the", "a", "an", "is", "are", "of", "to", "in", "and", "or",
        "for", "with", "on", "it", "this", "that", "what", "who",
        "how", "why", "does", "do", "did", "be", "as", "by", "from"
    }

    return {t for t in terms if t not in stopwords and len(t) > 1}


def is_definition_query(query: str) -> bool:
    q = normalize_text(query)
    return any(trigger in q for trigger in [
        "what is",
        "what are",
        "define",
        "explain",
        "meaning of",
        "describe"
    ])


def is_comparison_query(query: str) -> bool:
    q = normalize_text(query)
    return any(trigger in q for trigger in [
        "difference",
        "compare",
        "versus",
        "vs",
        "distinguish"
    ])


def is_summary_query(query: str) -> bool:
    q = normalize_text(query)
    return any(trigger in q for trigger in [
        "summarize",
        "summary",
        "overview",
        "main concepts",
        "key concepts",
        "core concepts"
    ])


def choose_top_k(query: str) -> int:
    if is_summary_query(query):
        return 10
    if is_comparison_query(query):
        return 8
    if is_definition_query(query):
        return 4
    return 6


def rewrite_query(query: str) -> str:
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

    # LangGraph concept rewrites
    if q in {"langgraph state", "state"}:
        return "state in langgraph definition application memory shared data structure"

    if q in {"stategraph", "langgraph stategraph"}:
        return "stategraph langgraph class build compile graph nodes edges state"

    if q in {"conditional edges", "langgraph conditional edges"}:
        return "conditional edges decide next node conditions logic current state"

    if q in {"toolnode", "tool node"}:
        return "toolnode special node run tool output state"

    if q in {"nodes", "langgraph nodes"}:
        return "nodes individual functions operations tasks graph state input output"

    if q in {"edges", "langgraph edges"}:
        return "edges connections nodes flow execution next node"

    if q in {"tools", "langgraph tools"}:
        return "tools specialized functions utilities nodes use api"

    if q in {"graph", "langgraph graph"}:
        return "graph structure maps tasks nodes connected executed workflow"

    if q in {"runnable", "langgraph runnable"}:
        return "runnable executable component task ai workflow modular systems"

    if "difference" in q and "state" in q and "stategraph" in q:
        return "state shared data structure application memory stategraph class build compile graph nodes edges"

    if "difference" in q and "nodes" in q and "edges" in q:
        return "nodes individual functions operations edges connections determine flow execution"

    if "difference" in q and "tools" in q and "toolnode" in q:
        return "tools specialized functions utilities toolnode special node run tool output state"

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
    if not pages:
        return False

    short_pages = sum(1 for p in pages if len(p["text"]) < 1200)
    return (short_pages / len(pages)) >= 0.7


# ─────────────────────────────────────────────────────────────
# Chroma ingestion + document listing
# ─────────────────────────────────────────────────────────────

def clear_collection_for_file(filename: str) -> None:
    try:
        existing = collection.get(where={"source": filename})
        ids = existing.get("ids", [])

        if ids:
            collection.delete(ids=ids)

    except Exception:
        pass


def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    start_time = time.time()

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
            title_clean = clean_title(title)

            docs.append(chunk)
            metadatas.append({
                "source": filename,
                "page": page["page_number"],
                "chunk_index": idx,
                "title": title,
                "title_clean": title_clean,
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
        "mode": "page-level" if slide_mode else "character-chunked",
        "ingestion_seconds": round(time.time() - start_time, 3)
    }


def list_documents() -> Dict[str, Any]:
    """
    List all unique documents currently stored in Chroma.
    """
    try:
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])

        docs = {}

        for meta in metadatas:
            source = meta.get("source", "Unknown")
            page = meta.get("page")

            if source not in docs:
                docs[source] = {
                    "doc_name": source,
                    "pages": set(),
                    "chunks": 0,
                    "doc_type": meta.get("doc_type", "unknown")
                }

            docs[source]["chunks"] += 1

            if page:
                docs[source]["pages"].add(page)

        result = []

        for doc in docs.values():
            result.append({
                "doc_name": doc["doc_name"],
                "pages": sorted(list(doc["pages"])),
                "page_count": len(doc["pages"]),
                "chunks": doc["chunks"],
                "doc_type": doc["doc_type"]
            })

        return {
            "documents": sorted(result, key=lambda x: x["doc_name"]),
            "document_count": len(result)
        }

    except Exception as e:
        return {
            "documents": [],
            "document_count": 0,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────────
# Retrieval + hybrid reranking
# ─────────────────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    top_k: Optional[int] = None,
    doc_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = choose_top_k(query)

    rewritten_query = rewrite_query(query)
    query_embedding = embedding_model.encode([rewritten_query]).tolist()

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }

    if doc_name:
        query_kwargs["where"] = {"source": doc_name}

    results = collection.query(**query_kwargs)

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


def keyword_score(query: str, text: str, title: str = "") -> float:
    """
    Keyword scorer used after vector retrieval:
    - boosts exact phrase matches
    - boosts title matches
    - rewards term overlap
    """
    raw_query_clean = clean_for_matching(query.replace("?", ""))
    rewritten_query_clean = clean_for_matching(rewrite_query(query))

    text_clean = clean_for_matching(text)
    title_clean = clean_title(title)

    raw_terms = get_terms(query)
    rewritten_terms = get_terms(rewrite_query(query))
    title_terms = get_terms(title_clean)
    text_terms = get_terms(text_clean)

    score = 0.0

    # Exact phrase matches
    if raw_query_clean and raw_query_clean == title_clean:
        score += 10.0
    elif raw_query_clean and raw_query_clean in title_clean:
        score += 8.0

    if raw_query_clean and raw_query_clean in text_clean:
        score += 5.0

    if rewritten_query_clean and rewritten_query_clean in title_clean:
        score += 6.0

    if rewritten_query_clean and rewritten_query_clean in text_clean:
        score += 4.0

    # Term overlap scoring
    score += len(raw_terms.intersection(title_terms)) * 2.5
    score += len(raw_terms.intersection(text_terms)) * 0.5
    score += len(rewritten_terms.intersection(title_terms)) * 2.0
    score += len(rewritten_terms.intersection(text_terms)) * 0.4

    return round(score, 4)


def concept_boost_score(query: str, title: str) -> float:
    """
    Domain-specific boosts for this LangGraph learning deck.
    These boosts make exact concept pages win over semantically related pages.
    """
    q_rewritten = normalize_text(rewrite_query(query))
    q_terms = get_terms(q_rewritten)
    title_clean = clean_title(title)

    boost = 0.0

    if "state" in q_terms and title_clean == "state":
        boost += 8.0

    if "stategraph" in q_terms and title_clean == "stategraph":
        boost += 8.0

    if "conditional" in q_terms and "edges" in q_terms and title_clean == "conditional edges":
        boost += 8.0

    if "toolnode" in q_terms and title_clean == "toolnode":
        boost += 8.0

    if "nodes" in q_terms and title_clean == "nodes":
        boost += 8.0

    if "edges" in q_terms and title_clean == "edges":
        boost += 8.0

    if "tools" in q_terms and title_clean == "tools":
        boost += 8.0

    if "graph" in q_terms and title_clean == "graph":
        boost += 8.0

    if "runnable" in q_terms and title_clean == "runnable":
        boost += 8.0

    return round(boost, 4)


def simple_rerank(query: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hybrid reranker:
    - semantic score from vector distance
    - keyword score from exact/lexical matching
    - concept-specific boosts
    """
    for item in retrieved:
        title_raw = item["metadata"].get("title", "")
        distance = item["distance"]

        semantic_score = max(0.0, 2.0 - distance)

        lexical_score = keyword_score(
            query=query,
            text=item["text"],
            title=title_raw
        )

        concept_boost = concept_boost_score(
            query=query,
            title=title_raw
        )

        final_score = (
            semantic_score * SEMANTIC_WEIGHT +
            lexical_score * KEYWORD_WEIGHT +
            concept_boost * CONCEPT_BOOST_WEIGHT
        )

        item["semantic_score"] = round(semantic_score, 4)
        item["keyword_score"] = round(lexical_score, 4)
        item["concept_boost"] = round(concept_boost, 4)
        item["rerank_score"] = round(final_score, 4)

    return sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)


def select_context_chunks(
    reranked_chunks: List[Dict[str, Any]],
    query: str,
    max_context_chunks: int = 3
) -> List[Dict[str, Any]]:
    if not reranked_chunks:
        return []

    if is_comparison_query(query):
        selected = [reranked_chunks[0]]
        top_score = reranked_chunks[0].get("rerank_score", 0)

        for chunk in reranked_chunks[1:max_context_chunks]:
            chunk_score = chunk.get("rerank_score", 0)

            if top_score > 0 and chunk_score >= top_score * 0.60:
                selected.append(chunk)

        return selected

    if is_definition_query(query):
        return [reranked_chunks[0]]

    if is_summary_query(query):
        return reranked_chunks[:max_context_chunks]

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
        title = meta.get("title", "Untitled")
        page = meta.get("page", "?")
        source = meta.get("source", "Unknown")

        sources.append(
            f'{source} page {page} (title: {title})'
        )

    return list(dict.fromkeys(sources))


def should_return_no_answer(selected_chunks: List[Dict[str, Any]]) -> bool:
    if not selected_chunks:
        return True

    top_score = selected_chunks[0].get("rerank_score", 0)
    return top_score < NO_ANSWER_SCORE_THRESHOLD


def get_source_names(chunks: List[Dict[str, Any]]) -> List[str]:
    return list(dict.fromkeys([
        item["metadata"].get("source", "Unknown")
        for item in chunks
    ]))


# ─────────────────────────────────────────────────────────────
# RAG answer generation
# ─────────────────────────────────────────────────────────────

def answer_with_rag(
    query: str,
    top_k: Optional[int] = None,
    max_context_chunks: int = 3,
    doc_name: Optional[str] = None
) -> Dict[str, Any]:
    start_time = time.time()

    effective_top_k = top_k or choose_top_k(query)

    retrieved = retrieve_chunks(
        query=query,
        top_k=effective_top_k,
        doc_name=doc_name
    )

    reranked = simple_rerank(query, retrieved)

    selected_chunks = select_context_chunks(
        reranked_chunks=reranked,
        query=query,
        max_context_chunks=max_context_chunks
    )

    selected_sources = build_source_list(selected_chunks)

    if should_return_no_answer(selected_chunks):
        return {
            "answer": "I could not find that in the uploaded document.",
            "sources": [],
            "retrieved_chunks": retrieved,
            "reranked_chunks": reranked[:max_context_chunks],
            "selected_context_chunks": [],
            "metrics": {
                "top_k": effective_top_k,
                "retrieved_count": len(retrieved),
                "selected_count": 0,
                "response_seconds": round(time.time() - start_time, 3),
                "no_answer_reason": "low_relevance_score",
                "no_answer_threshold": NO_ANSWER_SCORE_THRESHOLD,
                "doc_filter": doc_name or "all_documents",
                "retrieved_sources": get_source_names(retrieved),
                "selected_sources": [],
                "hybrid_scoring": {
                    "semantic_weight": SEMANTIC_WEIGHT,
                    "keyword_weight": KEYWORD_WEIGHT,
                    "concept_boost_weight": CONCEPT_BOOST_WEIGHT
                }
            }
        }

    context = build_context(selected_chunks)

    prompt = f"""
You are a precise knowledge assistant.

Use ONLY the supplied context.
If the answer is not present in the context, say exactly:
"I could not find that in the uploaded document."

Rules:
- Be concise and accurate.
- Use exact document meaning, not outside knowledge.
- For comparisons, clearly distinguish the concepts.
- For summaries, use bullet points.
- Do not cite sources that are not in the supplied context.
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

    valid_sources = set(selected_sources)
    sources = [s for s in sources if s in valid_sources]

    if not sources:
        sources = selected_sources

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
        "reranked_chunks": reranked[:max_context_chunks],
        "selected_context_chunks": selected_chunks,
        "metrics": {
            "top_k": effective_top_k,
            "retrieved_count": len(retrieved),
            "selected_count": len(selected_chunks),
            "response_seconds": round(time.time() - start_time, 3),
            "no_answer_threshold": NO_ANSWER_SCORE_THRESHOLD,
            "doc_filter": doc_name or "all_documents",
            "retrieved_sources": get_source_names(retrieved),
            "selected_sources": get_source_names(selected_chunks),
            "hybrid_scoring": {
                "semantic_weight": SEMANTIC_WEIGHT,
                "keyword_weight": KEYWORD_WEIGHT,
                "concept_boost_weight": CONCEPT_BOOST_WEIGHT
            }
        }
    }
