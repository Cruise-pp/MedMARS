import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pathlib import Path
from transformers import logging

logging.set_verbosity_error()

_model = None
_index = None
_corpus = None
_bm25 = None

_BASE_DIR = Path(__file__).resolve().parent.parent  # → 项目根目录

def _ensure_loaded():
    global _model, _index, _corpus, _bm25
    if _model is not None:
        return

    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _index = faiss.read_index(str(_BASE_DIR / "processed/medquad/medquad_index.faiss"))

    _corpus = []
    with open(_BASE_DIR / "processed/medquad/medquad_corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            _corpus.append(json.loads(line))

    tokenized_corpus = [
        doc["question"].lower().split() + str(doc.get("answer") or "").lower().split()
        for doc in _corpus
    ]
    _bm25 = BM25Okapi(tokenized_corpus)


def _compute_rrf(vector_ranks, bm25_ranks, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(vector_ranks):
        doc_id = int(doc_id)  # normalize numpy int → python int
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ranks):
        doc_id = int(doc_id)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def search(query: str, top_k: int = 5, retrieve_k: int = 50) -> list[dict]:
    _ensure_loaded()

    # ---- vector retrieval (FAISS) ----
    q_vec = _model.encode([query]) # type: ignore
    faiss.normalize_L2(q_vec)
    _, vector_indices = _index.search(q_vec, retrieve_k) # type: ignore
    vector_hits = vector_indices[0].tolist()

    # ---- keyword retrieval (BM25) ----
    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query) # type: ignore
    bm25_hits = np.argsort(bm25_scores)[::-1][:retrieve_k].tolist()

    # ---- RRF fusion ----
    fused_results = _compute_rrf(vector_hits, bm25_hits)

    results = []
    for doc_id, rrf_score in fused_results[:top_k]:
        if doc_id < 0 or doc_id >= len(_corpus): # type: ignore
            continue
        entry = _corpus[doc_id] # type: ignore
        results.append({
            "question": entry.get("question", ""),
            "answer": str(entry.get("answer") or ""),
            "score": float(rrf_score),
        })
    return results

if __name__ == "__main__":
    test_queries = [
        "What is type 2 diabetes?",
        "What are the side effects of metformin?",
        "How is hypertension treated?",
        "What causes chest pain?",
        "asdfghjkl",  # 垃圾输入，看 graceful degradation
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        results = search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.4f}")
            print(f"      Q: {r['question'][:100]}")
            print(f"      A: {r['answer'][:120]}...")