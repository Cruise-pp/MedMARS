import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict, Tuple

def strip_ns(tag: str) -> str:
    """Remove XML namespace: '{ns}tag' -> 'tag'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag

# 下面两个用于过滤明显不像“别名”的 synonym（可选）
_ec_like = re.compile(r"^\d+(\.\d+)+$")  # e.g., 3.4.21.5

def keep_synonym(raw: str) -> bool:
    if not raw:
        return False
    t = raw.strip()
    if len(t) < 2:
        return False
    if t.isdigit():
        return False
    if _ec_like.match(t):
        return False
    return True

def get_primary_drug_id(drug_elem: ET.Element) -> Optional[str]:
    """Find <drugbank-id primary='true'>DBxxxxx</drugbank-id>"""
    for ch in drug_elem.iter():
        if strip_ns(ch.tag) == "drugbank-id" and ch.attrib.get("primary") == "true":
            v = (ch.text or "").strip()
            return v or None
    return None

def get_drug_name(drug_elem: ET.Element) -> str:
    """Find first <name>...</name> under this drug."""
    for ch in drug_elem.iter():
        if strip_ns(ch.tag) == "name":
            return (ch.text or "").strip()
    return ""

def collect_synonyms(drug_elem: ET.Element, limit: Optional[int] = None) -> List[str]:
    """Collect all <synonym> text under this drug."""
    syns = []
    for ch in drug_elem.iter():
        if strip_ns(ch.tag) == "synonym":
            t = (ch.text or "").strip()
            if keep_synonym(t):
                syns.append(t)
                if limit is not None and len(syns) >= limit:
                    break
    return syns

def collect_drug_interactions(drug_elem: ET.Element) -> List[Dict]:
    """
    Collect all <drug-interaction> entries under this drug.
    Each interaction typically has:
      - <drugbank-id>DBxxxxx</drugbank-id>
      - <name>...</name>
      - <description>...</description>
    """
    interactions = []

    for inter in drug_elem.iter():
        if strip_ns(inter.tag) != "drug-interaction":
            continue

        dst_id = None
        dst_name = ""
        desc = ""

        for ch in inter:
            t = strip_ns(ch.tag)
            if t == "drugbank-id":
                dst_id = (ch.text or "").strip() or None
            elif t == "name":
                dst_name = (ch.text or "").strip()
            elif t == "description":
                desc = (ch.text or "").strip()

        if dst_id and desc:
            interactions.append({
                "dst_drug_id": dst_id,
                "dst_name": dst_name,
                "description": desc
            })

    return interactions

def parse_one_drug(drug_elem: ET.Element) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Returns:
      node_rec or None
      edge_recs: list of edges (src->dst with description)
    """
    src_id = get_primary_drug_id(drug_elem)
    if not src_id:
        return None, []

    src_name = get_drug_name(drug_elem)
    syns = collect_synonyms(drug_elem)

    node_rec = {
        "drug_id": src_id,
        "name": src_name,
        "synonyms": syns
    }

    edges = []
    for it in collect_drug_interactions(drug_elem):
        edges.append({
            "src_drug_id": src_id,
            "src_name": src_name,
            "dst_drug_id": it["dst_drug_id"],
            "dst_name": it["dst_name"],
            "description": it["description"]
        })

    return node_rec, edges

def extract_jsonl(xml_path: Path, out_nodes: Path, out_edges: Path, max_drugs: Optional[int] = None):
    out_nodes.parent.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)

    n_drugs_seen = 0
    n_nodes_written = 0
    n_edges_written = 0

    # stack 用于判断 “当前 drug 是否为顶层 drug（drugbank -> drug）”
    stack = []

    with out_nodes.open("w", encoding="utf-8") as fnodes, out_edges.open("w", encoding="utf-8") as fedges:
        for event, elem in ET.iterparse(xml_path, events=("start", "end")):
            if event == "start":
                stack.append(strip_ns(elem.tag))
                continue

            tag = strip_ns(elem.tag)

            # 只处理顶层 drug: ... <drugbank> <drug> ... </drug> </drugbank>
            if tag == "drug" and len(stack) >= 2 and stack[-2] == "drugbank":
                n_drugs_seen += 1

                node_rec, edge_recs = parse_one_drug(elem)

                if node_rec and node_rec.get("drug_id") and node_rec.get("name"):
                    fnodes.write(json.dumps(node_rec, ensure_ascii=False) + "\n")
                    n_nodes_written += 1

                for e in edge_recs:
                    fedges.write(json.dumps(e, ensure_ascii=False) + "\n")
                    n_edges_written += 1

                # 释放内存
                elem.clear()

                if max_drugs is not None and n_drugs_seen >= max_drugs:
                    break

                if n_drugs_seen % 500 == 0:
                    print(f"[progress] drugs_seen={n_drugs_seen} nodes={n_nodes_written} edges={n_edges_written}")

            # pop stack for end event
            stack.pop()

    print("[done]")
    print("drugs_seen:", n_drugs_seen)
    print("nodes_written:", n_nodes_written)
    print("edges_written:", n_edges_written)
    print("nodes_out:", out_nodes.resolve())
    print("edges_out:", out_edges.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="Datasets/drugbank.xml", help="Path to DrugBank XML")
    ap.add_argument("--out_dir", type=str, default="processed/drugbank", help="Output directory")
    ap.add_argument("--max_drugs", type=int, default=None, help="Optional cap for quick test")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    out_dir = Path(args.out_dir)

    out_nodes = out_dir / "drug_nodes.jsonl"
    out_edges = out_dir / "ddi_edges.jsonl"

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    extract_jsonl(xml_path, out_nodes, out_edges, max_drugs=args.max_drugs)

if __name__ == "__main__":
    main()