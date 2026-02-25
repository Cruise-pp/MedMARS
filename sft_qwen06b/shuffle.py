import json
import argparse
import random
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    random.shuffle(rows)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote shuffled file: {args.out_jsonl} | n={len(rows)} | seed={args.seed}")

if __name__ == "__main__":
    main()