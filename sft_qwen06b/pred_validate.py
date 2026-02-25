import os
import json
import argparse
from typing import Optional, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPT_TMPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n"
)

def build_prompt(ex: Dict) -> str:
    return PROMPT_TMPL.format(instruction=ex["instruction"], input=ex["input"])

def extract_json_obj(text: str) -> Optional[str]:
    t = (text or "").strip()
    i = t.find("{")
    j = t.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    return t[i:j+1].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--gt_jsonl", required=True, help="Use a fixed gt file")
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    if not os.path.exists(args.adapter_dir):
        raise FileNotFoundError(args.adapter_dir)
    if not os.path.exists(args.gt_jsonl):
        raise FileNotFoundError(args.gt_jsonl)

    # Load gt dataset
    ds = load_dataset("json", data_files={"gt": args.gt_jsonl})["gt"]
    n = min(args.n, len(ds))
    ds = ds.select(range(n))

    # Add gt_index = row number in this gt file (0..)
    ds = ds.add_column("gt_index", list(range(len(ds))))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = torch.bfloat16 if use_cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    os.makedirs(os.path.dirname(args.out_pred) or ".", exist_ok=True)

    with open(args.out_pred, "w", encoding="utf-8") as fout:
        for i in range(n):
            ex = ds[i]
            prompt = build_prompt(ex)

            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            if use_cuda:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )

            gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            raw = tok.decode(gen_ids, skip_special_tokens=True)

            cand = extract_json_obj(raw)
            parsed = None
            parse_ok = False
            if cand is not None:
                try:
                    parsed = json.loads(cand)
                    parse_ok = isinstance(parsed, dict)
                except Exception:
                    parsed = None
                    parse_ok = False

            record = {
                "gt_index": ex["gt_index"],
                "parse_ok": parse_ok,
                "pred_json": parsed,
                "raw": raw,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] wrote preds: {args.out_pred} | n={n} | gt={args.gt_jsonl}")

if __name__ == "__main__":
    main()