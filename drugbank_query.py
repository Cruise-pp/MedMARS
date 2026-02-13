# drugbank_query.py
import sqlite3
import re
import html
import difflib
from typing import List, Dict, Any, Tuple, Optional

DB_PATH = "processed/drugbank/drugbank_ddi.sqlite"

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_SPACES = re.compile(r"\s+")

# alias key cache: db_path -> List[str]
_ALIAS_KEYS_CACHE: Dict[str, List[str]] = {}


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def normalize_alias(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s


def _get_alias_keys(db_path: str) -> List[str]:
    if db_path in _ALIAS_KEYS_CACHE:
        return _ALIAS_KEYS_CACHE[db_path]

    conn = get_connection(db_path)
    try:
        rows = fetch_all(conn, "SELECT DISTINCT alias FROM aliases", ())
    finally:
        conn.close()

    keys: List[str] = []
    for r in rows:
        a = (r.get("alias") or "").strip()
        if a:
            keys.append(a)

    _ALIAS_KEYS_CACHE[db_path] = keys
    return keys

def _resolve_exact(query: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    norm = normalize_alias(query)
    if not norm:
        return {"query": query, "normalized": norm, "mode": "exact", "status": "not_found", "candidates": []}

    sql = """
    SELECT
      d.drug_id,
      d.name,
      d.degree,
      GROUP_CONCAT(DISTINCT a.source) AS source
    FROM aliases a
    JOIN drugs d ON d.drug_id = a.drug_id
    WHERE a.alias = ?
    GROUP BY d.drug_id, d.name, d.degree
    ORDER BY d.degree DESC, d.name ASC
    LIMIT ?
    """

    conn = get_connection(db_path)
    try:
        rows = fetch_all(conn, sql, (norm, int(top_n)))
    finally:
        conn.close()

    candidates = []
    for r in rows:
        candidates.append({
            "drug_id": (r.get("drug_id") or "").strip(),
            "name": (r.get("name") or "").strip(),
            "degree": int(r.get("degree") or 0),
            "source": (r.get("source") or "").strip(),
        })

    status = "not_found" if not candidates else ("unique" if len(candidates) == 1 else "candidates")
    return {"query": query, "normalized": norm, "mode": "exact", "status": status, "candidates": candidates}


def _resolve_prefix(query: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    norm = normalize_alias(query)
    if not norm:
        return {"query": query, "normalized": norm, "mode": "prefix", "status": "not_found", "candidates": []}

    # 用 alias 前缀做候选（更适合 Aspiri 这种）
    sql = """
    SELECT
      d.drug_id,
      d.name,
      d.degree,
      GROUP_CONCAT(DISTINCT a.source) AS source
    FROM aliases a
    JOIN drugs d ON d.drug_id = a.drug_id
    WHERE a.alias LIKE (? || '%')
    GROUP BY d.drug_id, d.name, d.degree
    ORDER BY d.degree DESC, d.name ASC
    LIMIT ?
    """

    conn = get_connection(db_path)
    try:
        rows = fetch_all(conn, sql, (norm, int(top_n)))
    finally:
        conn.close()

    candidates = [{
        "drug_id": (r.get("drug_id") or "").strip(),
        "name": (r.get("name") or "").strip(),
        "degree": int(r.get("degree") or 0),
        "source": (r.get("source") or "").strip(),
    } for r in rows]

    status = "not_found" if not candidates else ("unique" if len(candidates) == 1 else "candidates")
    return {"query": query, "normalized": norm, "mode": "prefix", "status": status, "candidates": candidates}

def _resolve_fuzzy(
    query: str,
    top_n: int = 10,
    db_path: str = DB_PATH,
    min_score: float = 0.8,
) -> Dict[str, Any]:
    norm = normalize_alias(query)
    if not norm:
        return {"query": query, "normalized": norm, "mode": "fuzzy", "status": "not_found", "candidates": []}

    alias_keys = _get_alias_keys(db_path)
    close_aliases = difflib.get_close_matches(
        norm,
        alias_keys,
        n=max(top_n * 5, 20),
        cutoff=float(min_score),
    )
    if not close_aliases:
        return {"query": query, "normalized": norm, "mode": "fuzzy", "status": "not_found", "candidates": []}

    # 查 matched alias 对应的 drug 候选（并对同 drug_id 取 best score）
    sql = """
    SELECT
      d.drug_id,
      d.name,
      d.degree,
      GROUP_CONCAT(DISTINCT a.source) AS source
    FROM aliases a
    JOIN drugs d ON d.drug_id = a.drug_id
    WHERE a.alias = ?
    GROUP BY d.drug_id, d.name, d.degree
    """

    conn = get_connection(db_path)
    try:
        best_by_drug: Dict[str, Dict[str, Any]] = {}
        for a in close_aliases:
            score = difflib.SequenceMatcher(None, norm, a).ratio()
            if score < float(min_score):
                continue

            rows = fetch_all(conn, sql, (a,))
            for r in rows:
                did = (r.get("drug_id") or "").strip()
                if not did:
                    continue
                cand = {
                    "drug_id": did,
                    "name": (r.get("name") or "").strip(),
                    "degree": int(r.get("degree") or 0),
                    "source": (r.get("source") or "").strip(),
                    "matched_alias": a,
                    "score": float(score),
                }
                prev = best_by_drug.get(did)
                if (prev is None) or (cand["score"] > prev["score"]):
                    best_by_drug[did] = cand
    finally:
        conn.close()

    candidates = list(best_by_drug.values())
    candidates.sort(key=lambda x: (-x["score"], -x["degree"], x["name"]))
    candidates = candidates[: int(top_n)]

    status = "not_found" if not candidates else ("unique" if len(candidates) == 1 else "candidates")
    return {"query": query, "normalized": norm, "mode": "fuzzy", "status": status, "candidates": candidates}


def resolve(
    query: str,
    top_n: int = 10,
    db_path: str = DB_PATH,
    allow_fuzzy: bool = True,
    min_score: float = 0.80,
    prefix_min_len: int = 6,
) -> Dict[str, Any]:
    # 1) exact
    r = _resolve_exact(query, top_n=top_n, db_path=db_path)
    if r["status"] != "not_found":
        return r

    # 2) prefix（短输入：Aspiri）
    norm = r.get("normalized") or ""
    if norm and len(norm) < int(prefix_min_len):
        rp = _resolve_prefix(query, top_n=top_n, db_path=db_path)
        if rp["status"] != "not_found":
            return rp

    # 3) fuzzy（typo：Bihydroergotamine）
    if allow_fuzzy:
        return _resolve_fuzzy(query, top_n=top_n, db_path=db_path, min_score=min_score)

    return r

def get_drug(drug_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    did = (drug_id or "").strip()
    if not did:
        return {"drug_id": drug_id, "status": "invalid", "drug": None}

    sql = """
    SELECT drug_id, name, indication, description
    FROM drugs
    WHERE drug_id = ?
    LIMIT 1
    """

    conn = get_connection(db_path)
    try:
        rows = fetch_all(conn, sql, (did,))
    finally:
        conn.close()

    if not rows:
        return {"drug_id": drug_id, "status": "not_found", "drug": None}
    return {"drug_id": did, "status": "found", "drug": rows[0]}


def neighbors(drug_id: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    did = (drug_id or "").strip()
    if not did:
        return {"drug_id": drug_id, "status": "invalid", "neighbors": []}

    sql = """
    SELECT
      e.dst_id AS neighbor_id,
      d.name   AS neighbor_name,
      e.description
    FROM ddi_edges e
    LEFT JOIN drugs d ON d.drug_id = e.dst_id
    WHERE e.src_id = ?
    LIMIT ?
    """

    conn = get_connection(db_path)
    try:
        rows = fetch_all(conn, sql, (did, int(top_n) * 50))
    finally:
        conn.close()

    seen = set()
    out = []
    for r in rows:
        nid = (r.get("neighbor_id") or "").strip()
        nname = (r.get("neighbor_name") or "").strip()
        desc = html.unescape((r.get("description") or "").strip())
        if not nid or not desc:
            continue
        key = (nid, desc)
        if key in seen:
            continue
        seen.add(key)
        out.append({"neighbor_id": nid, "neighbor_name": nname or nid, "description": desc})
        if len(out) >= int(top_n):
            break

    return {"drug_id": did, "status": ("found" if out else "not_found"), "neighbors": out}


def ddi_between(drug_id_a: str, drug_id_b: str, top_n: int = 20, db_path: str = DB_PATH) -> Dict[str, Any]:
    a = (drug_id_a or "").strip()
    b = (drug_id_b or "").strip()
    if not a or not b:
        return {"drug_a": drug_id_a, "drug_b": drug_id_b, "status": "invalid", "evidence": []}

    sql = """
    SELECT description
    FROM ddi_edges
    WHERE (src_id = ? AND dst_id = ?)
       OR (src_id = ? AND dst_id = ?)
    LIMIT ?
    """

    conn = get_connection(db_path)
    try:
        rows = fetch_all(conn, sql, (a, b, b, a, int(top_n)))
    finally:
        conn.close()

    # 去重 + unescape
    seen = set()
    ev = []
    for r in rows:
        d = html.unescape((r.get("description") or "").strip())
        if d and d not in seen:
            seen.add(d)
            ev.append(d)

    return {"drug_a": a, "drug_b": b, "status": ("found" if ev else "not_found"), "evidence": ev[: int(top_n)]}


def neighbors_by_name(query: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    r = resolve(query, top_n=top_n, db_path=db_path, allow_fuzzy=True)
    if r["status"] == "not_found":
        return {"query": query, 
                "normalized": r["normalized"], 
                "status": "not_found", 
                "candidates": [], 
                "neighbors": []
        }
    if r["status"] == "candidates":
        return {"query": query, 
                "normalized": r["normalized"], 
                "status": "candidates", 
                "candidates": r["candidates"], 
                "neighbors": []
        }

    did = r["candidates"][0]["drug_id"]
    nb = neighbors(did, top_n=top_n, db_path=db_path)
    return {"query": query, 
            "normalized": r["normalized"], 
            "status": "found", 
            "drug": r["candidates"][0], 
            "neighbors": nb["neighbors"]
    }


def ddi_between_by_name(drug_a: str, drug_b: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    ra = resolve(drug_a, top_n=top_n, db_path=db_path, allow_fuzzy=True)
    rb = resolve(drug_b, top_n=top_n, db_path=db_path, allow_fuzzy=True)

    if ra["status"] == "not_found" or rb["status"] == "not_found":
        return {"drug_a_query": drug_a, 
                "drug_b_query": drug_b, 
                "status": "not_found", 
                "a": ra, 
                "b": rb, 
                "evidence": []
        }

    if ra["status"] == "candidates" or rb["status"] == "candidates":
        return {"drug_a_query": drug_a, 
                "drug_b_query": drug_b, 
                "status": "candidates", 
                "a": ra, 
                "b": rb, 
                "evidence": []
        }

    a_id = ra["candidates"][0]["drug_id"]
    b_id = rb["candidates"][0]["drug_id"]
    ddi = ddi_between(a_id, b_id, top_n=top_n, db_path=db_path)
    return {"drug_a_query": drug_a, "drug_b_query": drug_b, "status": ddi["status"], "a": ra["candidates"][0], "b": rb["candidates"][0], "evidence": ddi["evidence"]}