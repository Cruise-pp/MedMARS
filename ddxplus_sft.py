"""
DDXPlus -> SFT (instruction/input/output) JSONL exporter (sampled).

What it does
------------
1) Loads DDXPlus metadata:
   - release_evidences.json (question_en, data_type, value_meaning, is_antecedent)
   - release_conditions.json (optional; not required for label output)
2) Reservoir-samples N rows from {split}.csv (streaming; avoids loading 1M rows into RAM)
3) Parses EVIDENCES tokens:
   - "E_91" -> present (binary)
   - "E_204_@_V_10" / "E_56_@_4" -> value-coded
4) Maps codes/values -> readable text + meaning_en
5) Compacts multi-choice evidence (same code appears multiple times)
6) Flattens into text with two sections:
   - Symptoms / Current findings
   - Antecedents / Risk factors
7) Writes JSONL with fields: instruction, input, output

Usage
-----
python ddxplus_sft_100.py --base_dir Datasets/DDXPlus --out processed/ddxplus_sft/train_100.jsonl --n 100 --seed 42

Notes
-----
- Output label follows PATHOLOGY as-is (e.g., "URTI").
- For some categorical questions, value_meaning uses "Y"/"N"; we normalize to "YES"/"NO" in the flattened text.
"""

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# -----------------------------
# Parsing utilities
# -----------------------------
def parse_list_cell(x: Any) -> List[Any]:
    """Parse a CSV cell that stores a list as a string."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    # Try JSON first (double quotes)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try Python literal (single quotes)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # Fallback split
    return [t.strip() for t in s.replace(",", ";").split(";") if t.strip()]


def parse_ddxplus_token(tok: Any) -> Optional[Tuple[str, bool, Optional[str]]]:
    """
    Parse DDXPlus token.
    - 'E_91' -> ('E_91', False, None)
    - 'E_204_@_V_10' -> ('E_204', True, 'V_10')
    - 'E_56_@_4' -> ('E_56', True, '4')
    """
    if tok is None or (isinstance(tok, float) and pd.isna(tok)):
        return None
    t = str(tok).strip().strip('"').strip("'")
    if not t:
        return None
    if "_@_" in t:
        code, val = t.split("_@_", 1)
        return code.strip(), True, val.strip()
    return t, False, None


def normalize_yesno(s: Any) -> Optional[str]:
    """Normalize variants of yes/no to YES/NO; return None if not yes/no."""
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"y", "yes", "true", "1", "oui"}:
        return "YES"
    if t in {"n", "no", "false", "0", "non"}:
        return "NO"
    return None


# -----------------------------
# Evidence normalization
# -----------------------------
def normalize_tokens_with_meta(tokens: List[Any], evid_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert raw tokens -> list of dict:
      {code, text, data_type, state, value?, meaning_en?}
    """
    out: List[Dict[str, Any]] = []
    for tok in tokens:
        parsed = parse_ddxplus_token(tok)
        if parsed is None:
            continue
        code, has_value, val = parsed

        meta = evid_dict.get(code, {})
        text = meta.get("question_en") or meta.get("name") or code
        dt = meta.get("data_type")
        vm = meta.get("value_meaning", {}) or {}

        ev: Dict[str, Any] = {
            "code": code,
            "text": text,
            "data_type": dt,
            "state": "value" if has_value else "present",
        }

        if has_value:
            ev["value"] = val
            meaning = vm.get(str(val))
            # In DDXPlus, meaning may be dict {"en":..., "fr":...}
            if isinstance(meaning, dict):
                if meaning.get("en") not in (None, ""):
                    ev["meaning_en"] = meaning.get("en")
            elif isinstance(meaning, str) and meaning.strip():
                ev["meaning_en"] = meaning.strip()

        out.append(ev)
    return out

# def normalize_topk(topk: List[Dict[str, Any]], key: str = "score") -> List[Dict[str, Any]]:
#     s = 0.0
#     for x in topk:
#         try:
#             s += float(x.get(key, 0.0))
#         except Exception:
#             pass
#     if s <= 0:
#         return topk
#     out = []
#     for x in topk:
#         y = dict(x)
#         y[key] = float(y.get(key, 0.0)) / s
#         out.append(y)
#     return out

def normalize_topk(topk, key="score", ndigits=2):
    s = 0.0
    for x in topk:
        try:
            s += float(x.get(key, 0.0))
        except Exception:
            pass
    if s <= 0:
        return topk

    out = []
    for x in topk:
        y = dict(x)
        try:
            y[key] = round(float(y.get(key, 0.0)) / s, ndigits)
        except Exception:
            y[key] = 0.0
        out.append(y)

    return out

def compact_evidence_list(norm_evs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge repeated codes (multi-choice):
      - B-type: present True/False
      - others: values list
    Output:
      {code,text,data_type,present,values:[{value, meaning_en?}, ...]}
    """
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for e in norm_evs:
        code = e["code"]
        if code not in merged:
            merged[code] = {
                "code": code,
                "text": e.get("text", code),
                "data_type": e.get("data_type"),
                "present": False,
                "values": [],
            }
            order.append(code)

        if e.get("state") == "present":
            merged[code]["present"] = True
        else:
            item = {"value": e.get("value")}
            if e.get("meaning_en") not in (None, ""):
                item["meaning_en"] = e.get("meaning_en")
            merged[code]["values"].append(item)

    return [merged[c] for c in order]


# -----------------------------
# Flattening (textification)
# -----------------------------
def render_compact_evidence(e: Dict[str, Any]) -> Optional[str]:
    """Render one compact evidence line; returns None if nothing to render."""
    text = e.get("text", e.get("code"))
    dt = e.get("data_type")
    present = bool(e.get("present", False))
    values = e.get("values", [])

    # Binary: only print if present True
    if dt == "B":
        return f"- {text} YES" if present else None

    # Non-binary: print if values exist
    if values:
        rendered: List[str] = []
        for item in values:
            meaning_en = item.get("meaning_en") if isinstance(item, dict) else None
            v = item.get("value") if isinstance(item, dict) else item

            yn = normalize_yesno(meaning_en)
            if yn is not None:
                rendered.append(yn)
            else:
                rendered.append(str(meaning_en) if meaning_en not in (None, "") else str(v))

        # de-dup keep order
        seen = set()
        uniq = []
        for r in rendered:
            if r not in seen:
                uniq.append(r)
                seen.add(r)

        # If effectively yes/no categorical, show single YES/NO
        if len(uniq) == 1 and uniq[0] in {"YES", "NO"}:
            return f"- {text} {uniq[0]}"
        return f"- {text} " + "; ".join(uniq)

    # Rare case: non-binary but present True
    if present:
        return f"- {text} YES"
    return None


def compact_evidence_to_text_split(compact_evs: List[Dict[str, Any]], evid_dict: Dict[str, Any],
                                  max_lines: Optional[int] = None) -> str:
    """Split evidence into Symptoms vs Antecedents using is_antecedent from evid_dict."""
    symptoms_lines: List[str] = []
    antecedent_lines: List[str] = []
    total = 0

    for e in compact_evs:
        code = e.get("code")
        is_ant = bool(evid_dict.get(code, {}).get("is_antecedent", False)) # type: ignore

        line = render_compact_evidence(e)
        if line is None:
            continue

        if is_ant:
            antecedent_lines.append(line)
        else:
            symptoms_lines.append(line)

        total += 1
        if max_lines is not None and total >= max_lines:
            break

    parts: List[str] = []
    if symptoms_lines:
        parts.append("Symptoms & Current findings:\n" + "\n".join(symptoms_lines))
    if antecedent_lines:
        parts.append("Antecedents & Risk factors:\n" + "\n".join(antecedent_lines))
    return "\n\n".join(parts)


def make_sft_example(row: Dict[str, Any], evid_dict: Dict[str, Any], max_lines: Optional[int] = None, k: int = 5) -> Dict[str, Any]:
    """Create a single SFT example from a sampled row dict."""
    age = row.get("AGE")
    sex = row.get("SEX")
    label = row.get("PATHOLOGY")

    ddx_raw = parse_list_cell(row.get("DIFFERENTIAL_DIAGNOSIS"))
    ddx_pairs = []
    for item in ddx_raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name, prob = item[0], item[1]
            try:
                ddx_pairs.append((str(name).strip(), round(float(prob), 3)))
            except Exception:
                continue

    ddx_pairs.sort(key=lambda x: x[1], reverse=True)
    topk = [{"label": n, "score": p} for n, p in ddx_pairs[:k]]
    topk = normalize_topk(topk)

    output_obj = {
        "primary_diagnosis": label,
        "differential_diagnosis": topk,
    }

    # initial evidence -> header line
    init_tok = row.get("INITIAL_EVIDENCE")
    init_parsed = parse_ddxplus_token(init_tok)
    init_text = None
    if init_parsed is not None:
        init_code, init_has_value, init_val = init_parsed
        meta = evid_dict.get(init_code, {})
        q = meta.get("question_en") or meta.get("name") or init_code
        init_text = f"Initial complaint: {q} (YES)"

    tokens = parse_list_cell(row.get("EVIDENCES"))
    norm_evs = normalize_tokens_with_meta(tokens, evid_dict)
    compact_evs = compact_evidence_list(norm_evs)
    findings = compact_evidence_to_text_split(compact_evs, evid_dict, max_lines=max_lines)

    header = [f"Patient: age={age}, sex={sex}"]
    if init_text:
        header.append(init_text)

    input_text = "\n".join(header) + "\n\n" + findings

    instruction = (
        "You are an expert in medical diagnostic reasoning."
        "Analyze the patient's demographics, initial complaint, and symptoms/antecedents/risk factors."
        "Provide a structered assessement in valid JSON format with two fields: "
        '"primary_diagnosis" (the single most likely diagnosis label) and '
        '"differential_diagnosis" (a list of top candidate diseases with their probabilities).'
        "Do not output any conversational text, only the JSON object."
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": json.dumps(output_obj, ensure_ascii=False)
    }


# -----------------------------
# Sampling + Export
# -----------------------------
def _normalize_colname(c: str) -> str:
    return str(c).strip().lstrip("\ufeff")

def reservoir_sample_csv(csv_path: Path, n: int, seed: int, chunksize: int = 50000) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    required = ["AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"]
    optional = ["DIFFERENTIAL_DIAGNOSIS"]

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk.rename(columns=_normalize_colname)

        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}\nActual cols={list(chunk.columns)}")

        cols = required + [c for c in optional if c in chunk.columns]
        chunk = chunk[cols]

        for _, row in chunk.iterrows():
            row_dict = {c: row[c] for c in cols}

            if seen < n:
                reservoir.append(row_dict)
            else:
                j = rng.randint(0, seen)
                if j < n:
                    reservoir[j] = row_dict
            seen += 1

    if seen == 0:
        raise RuntimeError(f"No rows read from {csv_path}")
    return reservoir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="Datasets/DDXPlus", help="DDXPlus directory")
    ap.add_argument("--split", type=str, default="validate", choices=["train", "validate", "test"], help="Which CSV split")
    ap.add_argument("--n", type=int, default=100, help="Number of samples to export")
    ap.add_argument("--k", type=int, default=5, help="Top-k size for differential diagnoses")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize for streaming read")
    ap.add_argument("--max_lines", type=int, default=None, help="Optional cap on total evidence lines")
    ap.add_argument("--out", type=str, default="processed/ddxplus_sft/validate.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    base = Path(args.base_dir)
    csv_path = base / f"{args.split}.csv"
    evid_path = base / "release_evidences.json"
    cond_path = base / "release_conditions.json"

    for p in [csv_path, evid_path, cond_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(evid_path, "r", encoding="utf-8") as f:
        evid_dict = json.load(f)

    # Loaded for future extension; not needed for label-only SFT
    with open(cond_path, "r", encoding="utf-8") as f:
        _ = json.load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    required = ["AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"]
    optional = ["DIFFERENTIAL_DIAGNOSIS"]

    def _norm(c: str) -> str:
        return str(c).strip().lstrip("\ufeff")
    
    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for chunk in pd.read_csv(csv_path, chunksize=args.chunksize):
            chunk = chunk.rename(columns=_norm)

            missing = [c for c in required if c not in chunk.columns]
            if missing:
                raise ValueError(
                    f"Missing required columns in {csv_path}: {missing}\nActual cols={list(chunk.columns)}"
            )
            cols = required + [c for c in optional if c in chunk.columns]

            chunk = chunk[cols]

            for row_vals in chunk.itertuples(index=False, name=None):
                row = dict(zip(cols, row_vals))
                ex = make_sft_example(row, evid_dict, max_lines=args.max_lines, k=args.k)
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} SFT examples to: {out_path.resolve()}")

    # sampled_rows = reservoir_sample_csv(csv_path, n=args.n, seed=args.seed, chunksize=args.chunksize)

    # out_path = Path(args.out)
    # out_path.parent.mkdir(parents=True, exist_ok=True)

    # with open(out_path, "w", encoding="utf-8") as fout:
    #     for row in sampled_rows:
    #         ex = make_sft_example(row, evid_dict, max_lines=args.max_lines, k=args.k)
    #         fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # print(f"Wrote {len(sampled_rows)} SFT examples to: {out_path.resolve()}")



if __name__ == "__main__":
    main()
