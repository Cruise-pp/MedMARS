"""
Retrieval Quality Evaluation
============================
Evaluates the retrieval components of the RAG pipeline WITHOUT calling any LLM API.

VectorRAG (MedQuAD):  Hit@K, MRR, NDCG@K
GraphRAG  (DrugBank): Resolution Accuracy, DDI Recall
"""

import json
import math
import random
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "processed"


# ================================================================
# Metric helpers
# ================================================================

def hit_at_k(gold_rank: Optional[int], k: int) -> int:
    """1 if gold document appears within top-K, else 0."""
    if gold_rank is None:
        return 0
    return 1 if gold_rank < k else 0


def reciprocal_rank(gold_rank: Optional[int]) -> float:
    """1/(rank+1) if found, else 0."""
    if gold_rank is None:
        return 0.0
    return 1.0 / (gold_rank + 1)


def ndcg_at_k(gold_rank: Optional[int], k: int) -> float:
    """NDCG@K with binary relevance (single gold document).

    Only one relevant doc exists, so IDCG = 1/log2(2) = 1.0.
    DCG  = 1/log2(gold_rank+2) if gold_rank < k, else 0.
    """
    if gold_rank is None or gold_rank >= k:
        return 0.0
    idcg = 1.0  # 1/log2(2)
    dcg = 1.0 / math.log2(gold_rank + 2)
    return dcg / idcg


# ================================================================
# VectorRAG Retrieval Evaluation
# ================================================================

def eval_vectorrag_retrieval(
    num_samples: int = 200,
    top_k: int = 10,
    report_k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, Any]:
    """Evaluate MedQuAD hybrid retrieval (FAISS + BM25 + RRF).

    Samples ``num_samples`` QA pairs from the corpus, uses each question
    as a query, and checks whether the gold document appears in the top-K
    retrieval results.
    """
    from medquad_rag import query_index as mq

    # Load corpus for gold matching
    corpus_path = PROCESSED_DIR / "medquad" / "medquad_corpus.jsonl"
    corpus = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    # Sample test queries
    samples = random.sample(corpus, min(num_samples, len(corpus)))

    gold_ranks = []
    for sample in tqdm(samples, desc="VectorRAG Retrieval Eval"):
        query = sample["question"]
        gold_question = sample["question"]

        results = mq.search(query, top_k=top_k)

        # Find gold document rank
        rank = None
        for i, r in enumerate(results):
            if r["question"].strip() == gold_question.strip():
                rank = i
                break
        gold_ranks.append(rank)

    # Compute metrics
    n = len(gold_ranks)
    metrics = {}

    for k in report_k_values:
        if k > top_k:
            continue
        metrics[f"Hit@{k}"] = sum(hit_at_k(r, k) for r in gold_ranks) / n

    metrics["MRR"] = sum(reciprocal_rank(r) for r in gold_ranks) / n

    for k in report_k_values:
        if k > top_k:
            continue
        metrics[f"NDCG@{k}"] = sum(ndcg_at_k(r, k) for r in gold_ranks) / n

    # How many were not found at all in top-K
    not_found = sum(1 for r in gold_ranks if r is None)
    metrics["not_found_in_top_k"] = not_found
    metrics["num_samples"] = n
    metrics["top_k"] = top_k

    return metrics


# ================================================================
# GraphRAG Retrieval Evaluation
# ================================================================

def eval_graphrag_resolution(
    num_samples: int = 200,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate DrugBank drug name resolution accuracy.

    Samples drug names from the database and checks whether resolve()
    maps them back to the correct drug_id.
    """
    from drugbank_graph import drugbank_query as dq

    if db_path is None:
        db_path = str(PROCESSED_DIR / "drugbank" / "drugbank_ddi.sqlite")

    # Sample drugs from database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT drug_id, name FROM drugs ORDER BY RANDOM() LIMIT ?", (num_samples,))
    drugs = [dict(r) for r in cur.fetchall()]
    conn.close()

    exact_correct = 0
    total = len(drugs)

    # Also test prefix and fuzzy by constructing variants
    prefix_correct = 0
    prefix_total = 0
    fuzzy_correct = 0
    fuzzy_total = 0

    for drug in tqdm(drugs, desc="GraphRAG Resolution Eval"):
        drug_id = drug["drug_id"]
        name = drug["name"]

        # --- Exact name test ---
        result = dq.resolve(name)
        if result["status"] != "not_found" and result["candidates"]:
            if result["candidates"][0]["drug_id"] == drug_id:
                exact_correct += 1

        # --- Prefix test (first 5 chars, only if name is long enough) ---
        if len(name) >= 8:
            prefix = name[:5]
            prefix_total += 1
            result_p = dq.resolve(prefix)
            if result_p["status"] != "not_found" and result_p["candidates"]:
                if any(c["drug_id"] == drug_id for c in result_p["candidates"]):
                    prefix_correct += 1

        # --- Fuzzy test (swap two adjacent chars to simulate typo) ---
        if len(name) >= 6:
            mid = len(name) // 2
            typo_name = name[:mid] + name[mid+1] + name[mid] + name[mid+2:]
            fuzzy_total += 1
            result_f = dq.resolve(typo_name)
            if result_f["status"] != "not_found" and result_f["candidates"]:
                if any(c["drug_id"] == drug_id for c in result_f["candidates"]):
                    fuzzy_correct += 1

    metrics = {
        "exact_accuracy": exact_correct / total if total else 0,
        "exact_correct": exact_correct,
        "exact_total": total,
        "prefix_accuracy": prefix_correct / prefix_total if prefix_total else 0,
        "prefix_correct": prefix_correct,
        "prefix_total": prefix_total,
        "fuzzy_accuracy": fuzzy_correct / fuzzy_total if fuzzy_total else 0,
        "fuzzy_correct": fuzzy_correct,
        "fuzzy_total": fuzzy_total,
    }
    return metrics


def eval_graphrag_ddi(
    num_samples: int = 200,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate DrugBank DDI retrieval recall.

    Samples known DDI pairs from the database and checks whether
    ddi_between() successfully retrieves the interaction evidence.
    """
    from drugbank_graph import drugbank_query as dq

    if db_path is None:
        db_path = str(PROCESSED_DIR / "drugbank" / "drugbank_ddi.sqlite")

    # Sample known DDI pairs
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT src_id, dst_id FROM ddi_edges ORDER BY RANDOM() LIMIT ?",
        (num_samples,)
    )
    pairs = [dict(r) for r in cur.fetchall()]
    conn.close()

    hit = 0
    total = len(pairs)

    for pair in tqdm(pairs, desc="GraphRAG DDI Recall Eval"):
        result = dq.ddi_between(pair["src_id"], pair["dst_id"])
        if result["status"] == "found" and len(result["evidence"]) > 0:
            hit += 1

    metrics = {
        "ddi_recall": hit / total if total else 0,
        "ddi_hit": hit,
        "ddi_total": total,
    }
    return metrics


# ================================================================
# Main
# ================================================================

def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")
    print()


if __name__ == "__main__":
    NUM_SAMPLES = 200

    # 1. VectorRAG Retrieval
    print("\n--- VectorRAG Retrieval Evaluation ---")
    vr_metrics = eval_vectorrag_retrieval(num_samples=NUM_SAMPLES, top_k=10)
    print_metrics("VectorRAG Retrieval (MedQuAD)", vr_metrics)

    # 2. GraphRAG Resolution Accuracy
    print("\n--- GraphRAG Resolution Evaluation ---")
    gr_metrics = eval_graphrag_resolution(num_samples=NUM_SAMPLES)
    print_metrics("GraphRAG Drug Resolution (DrugBank)", gr_metrics)

    # 3. GraphRAG DDI Recall
    print("\n--- GraphRAG DDI Recall Evaluation ---")
    ddi_metrics = eval_graphrag_ddi(num_samples=NUM_SAMPLES)
    print_metrics("GraphRAG DDI Retrieval (DrugBank)", ddi_metrics)
