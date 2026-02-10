import sqlite3
import re
import html
from typing import List, Dict, Any, Tuple, Optional

DB_PATH = "processed/drugbank/drugbank_ddi.sqlite"
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_SPACES = re.compile(r"\s+")

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with dict-like rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_all(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
    """Run a query and return rows as list[dict]."""
    cur = conn.cursor()
    cur.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]

def normalize_alias(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s

def resolve_exact(query: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    norm = normalize_alias(query)
    print(f"[INFO] Resolving exact: {query} -> {norm}")

    if not norm:
        return {"query": query, 
                "normalized": norm, 
                "mode": "exact", 
                "status": "not_found", 
                "candidates": []
                }

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
        candidates = fetch_all(conn, sql, (norm, int(top_n)))
    finally:
        conn.close()

    status = "not_found" if len(candidates) == 0 else ("unique" if len(candidates) == 1 else "candidates")

    return {
        "query": query, 
        "normalized": norm, 
        "mode": "exact", 
        "status": status, 
        "candidates": candidates
    }

def ddi_between(drug_id_a: str, drug_id_b: str, db_path: str = DB_PATH, top_n=20) -> Dict[str, Any]:
    a = (drug_id_a or "").strip()
    b = (drug_id_b or "").strip()

    if not a or not b:
        return {
            "drug_a": drug_id_a,
            "drug_b": drug_id_b,
            "status": "invalid",
            "edges": []
        }
    
    sql = """
    SELECT src_id, dst_id, description
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

    descs = [html.unescape((r.get("description") or "").strip()) for r in rows]
    descs = [d for d in descs if d]

    seen = set()
    evidence = []
    for d in descs:
        if d not in seen:
            seen.add(d)
            evidence.append(d)

    status = "found" if len(rows) > 0 else "not_found"

    return {"drug_a": a, 
            "drug_b": b, 
            "status": status, 
            "evidence": evidence[: int(top_n)]}

# drugbank_query.py (Block 4) — neighbors：给定一个 drug_id，返回 top-K 相互作用邻居 + 证据句（去重 + HTML unescape）

import html
from typing import List

def neighbors(drug_id: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    did = (drug_id or "").strip()
    if not did:
        return {"drug_id": drug_id, 
                "status": "invalid", 
                "neighbors": []
        }

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
        out.append({"neighbor_id": nid, 
                    "neighbor_name": nname or nid, 
                    "description": desc}
        )
        if len(out) >= int(top_n):
            break

    status = "found" if out else "not_found"
    return {"drug_id": did, 
            "status": status, 
            "neighbors": out
    }

def get_drug(drug_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    did = (drug_id or "").strip()
    if not did:
        return {
            "drug_id": drug_id,
            "status": "invalid",
            "drug": None
        }
    
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
        return {
            "drug_id": drug_id,
            "status": "not_found",
            "drug": None
        }
    return {
        "drug_id": drug_id,
        "status": "found",
        "drug": rows[0]
    }

def neighbors_by_name(query: str, top_n: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    r = resolve_exact(query, top_n=10, db_path=db_path)

    if r["status"] == "not_found":
        return {
            "query": query,
            "normalized": r["normalized"],
            "status": "not_found",
            "candidates": [],
            "neighbors": []
        }
    
    if r["status"] == "candidates":
        return {
            "query": query,
            "normalized": r["normalized"],
            "status": "candidates",
            "candidates": r["candidates"],
            "neighbors": []
        }
    
    did = r["candidates"][0]["drug_id"]
    nb = neighbors(did, top_n=top_n, db_path=db_path)
    return {
        "query": query,
        "normalized": r["normalized"],
        "status": "found",
        "candidates": r["candidates"],
        "drug": r["candidates"][0],
        "neighbors": nb["neighbors"]
    }

"""
TODO: logic for agent
"""