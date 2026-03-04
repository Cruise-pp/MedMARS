"""
Evaluate predictions from pred_validate.py.

Each line in pred_jsonl is self-contained:
{
  "gt":   {"primary_diagnosis": "...", "differential_diagnosis": [...]},
  "pred": {"parse_ok": bool, "pred_json": {...}, "raw": "..."}
}

No separate gt file needed.
"""

import json
import argparse


def load_jsonl(path, n=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help="Prediction JSONL from pred_validate.py")
    ap.add_argument("--k", type=int, default=5, help="Top-k for recall calculation")
    ap.add_argument("--n", type=int, default=None, help="Cap number of rows to evaluate")
    args = ap.parse_args()

    pred = load_jsonl(args.pred_jsonl, args.n)

    total = len(pred)
    parse_fail = 0
    used = 0
    acc1 = 0
    hitk = 0

    for r in pred:
        gt_block   = r.get("gt", {})
        pred_block = r.get("pred", {})

        gt_primary = (gt_block.get("primary_diagnosis") or "").strip()

        # check parse
        if not pred_block.get("parse_ok", False):
            parse_fail += 1
            continue

        pred_json = pred_block.get("pred_json")
        if not isinstance(pred_json, dict):
            parse_fail += 1
            continue

        used += 1

        # primary accuracy
        pred_primary = (pred_json.get("primary_diagnosis") or "").strip()
        if pred_primary == gt_primary:
            acc1 += 1

        # recall@k
        topk = pred_json.get("differential_diagnosis", [])
        labels = []
        if isinstance(topk, list):
            for item in topk[:args.k]:
                if isinstance(item, dict) and "label" in item:
                    labels.append(str(item["label"]).strip())

        if gt_primary in labels:
            hitk += 1

    print(f"[Eval] total_rows={total}")
    print(f"[Eval] used={used} | parse_fail={parse_fail}")

    if used == 0:
        print("No usable rows. Check pred_json fields in prediction file.")
        return

    print(f"Primary accuracy: {acc1 / used:.4f} ({acc1}/{used})")
    print(f"Recall@{args.k}:        {hitk / used:.4f} ({hitk}/{used})")


if __name__ == "__main__":
    main()