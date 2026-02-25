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


def parse_gt_primary(gt_row):
    """
    GT JSONL 每行: {"instruction":..., "input":..., "output":"{...json...}"}
    output 是 JSON 字符串（ground truth）
    """
    out = gt_row.get("output", "")
    obj = json.loads(out) if isinstance(out, str) else out
    gt_primary = (obj.get("primary_diagnosis") or "").strip()
    return gt_primary, obj


def parse_pred_obj(pred_row):
    """
    Pred JSONL: {"gt_index": int, "parse_ok": bool, "pred_json": dict|None, "raw": str}
    """
    obj = pred_row.get("pred_json", None)
    if not isinstance(obj, dict):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_jsonl", required=True, help="gt jsonl file")
    ap.add_argument("--pred_jsonl", required=True, help="Predictions jsonl produced by predict script")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=None, help="cap number of pred rows to evaluate")
    args = ap.parse_args()

    gt = load_jsonl(args.gt_jsonl, None)
    pred = load_jsonl(args.pred_jsonl, args.n)

    total = len(pred)
    used = 0
    pred_parse_fail = 0
    acc1 = 0
    hitk = 0

    for r in pred:
        gi = r.get("gt_index", None)
        if gi is None or not isinstance(gi, int) or gi < 0 or gi >= len(gt):
            continue

        gt_primary, _ = parse_gt_primary(gt[gi])

        pobj = parse_pred_obj(r)
        if pobj is None:
            pred_parse_fail += 1
            continue

        used += 1

        pred_primary = (pobj.get("primary_diagnosis") or "").strip()
        if pred_primary == gt_primary:
            acc1 += 1

        topk = pobj.get("differential_diagnosis", [])
        labels = []
        if isinstance(topk, list):
            for item in topk[: args.k]:
                if isinstance(item, dict) and "label" in item:
                    labels.append(str(item["label"]).strip())

        if gt_primary in labels:
            hitk += 1

    print(f"[Eval] total_pred_rows={total}")
    print(f"[Eval] used_for_metrics={used} | pred_parse_fail={pred_parse_fail}")
    if used == 0:
        print("No usable rows. Check gt_index/pred_json fields.")
        return

    print(f"Primary accuracy: {acc1/used:.3f} ({acc1}/{used})")
    print(f"Recall@{args.k}: {hitk/used:.3f} ({hitk}/{used})")


if __name__ == "__main__":
    main()