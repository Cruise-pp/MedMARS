"""
Build DrugBank DDI SQLite database from:
  - drug_nodes.jsonl  (drug_id, name, synonyms, indication[, description])
  - ddi_edges.jsonl   (src_drug_id, dst_drug_id, description[, src_name, dst_name])

No argparse. Edit the CONFIG section and run:
  python scripts/build_drugbank_sqlite_noargs.py
"""

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


# =========================
# CONFIG (edit these)
# =========================
NODES_JSONL = Path("../processed/drugbank/drug_nodes.jsonl")
EDGES_JSONL = Path("../processed/drugbank/ddi_edges.jsonl")
OUT_DB = Path("../processed/drugbank/drugbank_ddi.sqlite")

OVERWRITE = True
COMMIT_EVERY_NODES = 20000
COMMIT_EVERY_EDGES = 50000
# =========================


_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_SPACES = re.compile(r"\s+")
_EC_LIKE = re.compile(r"^\d+(\.\d+)+$")  # e.g., 3.4.21.5


def normalize_alias(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s


def keep_synonym(raw: str) -> bool:
    if not raw:
        return False
    t = raw.strip()
    if len(t) < 2:
        return False
    if t.isdigit():
        return False
    if _EC_LIKE.match(t):
        return False
    return True


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # speed pragmas (safe for one-time build)
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE drugs (
            drug_id TEXT PRIMARY KEY,
            name TEXT,
            indication TEXT,
            description TEXT,
            degree INTEGER DEFAULT 0
        );

        CREATE TABLE aliases (
            alias TEXT,
            drug_id TEXT,
            source TEXT,
            FOREIGN KEY(drug_id) REFERENCES drugs(drug_id)
        );

        CREATE TABLE ddi_edges (
            src_id TEXT,
            dst_id TEXT,
            src_name TEXT,
            dst_name TEXT,
            description TEXT
        );

        CREATE INDEX idx_alias ON aliases(alias);
        CREATE INDEX idx_alias_drug ON aliases(drug_id);

        CREATE INDEX idx_ddi_src_dst ON ddi_edges(src_id, dst_id);
        CREATE INDEX idx_ddi_src ON ddi_edges(src_id);
        CREATE INDEX idx_ddi_dst ON ddi_edges(dst_id);
        """
    )
    conn.commit()


def load_nodes(conn: sqlite3.Connection, nodes_path: Path) -> tuple[int, int]:
    cur = conn.cursor()
    n_drugs = 0
    n_alias = 0

    for obj in iter_jsonl(nodes_path):
        drug_id = (obj.get("drug_id") or "").strip()
        name = (obj.get("name") or "").strip()
        if not drug_id or not name:
            continue

        indication = (obj.get("indication") or "").strip()
        description = (obj.get("description") or "").strip()
        syns = obj.get("synonyms") or []

        cur.execute(
            "INSERT INTO drugs(drug_id, name, indication, description) VALUES (?, ?, ?, ?)",
            (drug_id, name, indication, description),
        )
        n_drugs += 1

        # alias from canonical name
        a = normalize_alias(name)
        if a:
            cur.execute(
                "INSERT INTO aliases(alias, drug_id, source) VALUES (?, ?, ?)",
                (a, drug_id, "name"),
            )
            n_alias += 1

        # alias from synonyms
        for s in syns:
            if not keep_synonym(s):
                continue
            a = normalize_alias(s)
            if not a:
                continue
            cur.execute(
                "INSERT INTO aliases(alias, drug_id, source) VALUES (?, ?, ?)",
                (a, drug_id, "synonym"),
            )
            n_alias += 1

        if n_drugs % COMMIT_EVERY_NODES == 0:
            conn.commit()
            print(f"[progress] nodes: drugs={n_drugs} aliases={n_alias}")

    conn.commit()
    print(f"[OK] nodes loaded: drugs={n_drugs} aliases={n_alias}")
    return n_drugs, n_alias


def load_edges_and_degree(conn: sqlite3.Connection, edges_path: Path) -> tuple[int, dict[str, int]]:
    cur = conn.cursor()
    deg = defaultdict(int)
    n_edges = 0

    for e in iter_jsonl(edges_path):
        src = (e.get("src_drug_id") or "").strip()
        dst = (e.get("dst_drug_id") or "").strip()
        desc = (e.get("description") or "").strip()
        if not src or not dst or not desc:
            continue

        cur.execute(
            "INSERT INTO ddi_edges(src_id, dst_id, src_name, dst_name, description) VALUES (?, ?, ?, ?, ?)",
            (src, dst, e.get("src_name", "") or "", e.get("dst_name", "") or "", desc),
        )

        deg[src] += 1
        deg[dst] += 1
        n_edges += 1

        if n_edges % COMMIT_EVERY_EDGES == 0:
            conn.commit()
            print(f"[progress] edges inserted: {n_edges}")

    conn.commit()
    print(f"[OK] edges loaded: {n_edges}")
    return n_edges, deg


def write_degree(conn: sqlite3.Connection, deg: dict[str, int]) -> None:
    cur = conn.cursor()
    updates = [(int(d), str(k).strip()) for k, d in deg.items()]
    before = conn.total_changes
    cur.executemany("UPDATE drugs SET degree=? WHERE drug_id=?", updates)
    conn.commit()
    changed = conn.total_changes - before
    print(f"[OK] degree updated rows (approx): {changed}")


# def sanity_check(conn: sqlite3.Connection) -> None:
#     cur = conn.cursor()

#     cur.execute("SELECT COUNT(*) FROM drugs")
#     drugs_cnt = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(*) FROM aliases")
#     alias_cnt = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(*) FROM ddi_edges")
#     edge_cnt = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(*) FROM drugs WHERE degree > 0")
#     deg_cnt = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(*) FROM drugs WHERE indication IS NOT NULL AND LENGTH(TRIM(indication)) > 0")
#     ind_cnt = cur.fetchone()[0]

#     print("[CHECK] drugs:", drugs_cnt)
#     print("[CHECK] aliases:", alias_cnt)
#     print("[CHECK] ddi_edges:", edge_cnt)
#     print("[CHECK] degree>0 drugs:", deg_cnt)
#     print("[CHECK] has indication:", ind_cnt)

#     cur.execute("SELECT drug_id, name, degree FROM drugs ORDER BY degree DESC LIMIT 5")
#     print("[CHECK] top degree:", cur.fetchall())


def main():
    if not NODES_JSONL.exists():
        raise FileNotFoundError(f"Missing nodes jsonl: {NODES_JSONL}")
    if not EDGES_JSONL.exists():
        raise FileNotFoundError(f"Missing edges jsonl: {EDGES_JSONL}")

    if OUT_DB.exists():
        if OVERWRITE:
            OUT_DB.unlink()
        else:
            raise FileExistsError(f"DB exists: {OUT_DB} (set OVERWRITE=True to replace)")

    conn = connect(OUT_DB)
    try:
        create_schema(conn)
        load_nodes(conn, NODES_JSONL)
        _, deg = load_edges_and_degree(conn, EDGES_JSONL)
        write_degree(conn, deg)
        # sanity_check(conn)
        print("[DONE] DB created at:", OUT_DB.resolve())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
