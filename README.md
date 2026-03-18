# MedMARS: Collaborative Multi-Agent Clinical Assistance via Multi-modal Reasoning

A multi-agent medical AI system built on **LangGraph** that combines fine-tuned language models, vision models, and retrieval-augmented generation (RAG) to assist with medical diagnosis, medication interaction queries, and general medical Q&A.

---

## Team
- Yufan Shi — Data Preprocessing/ Agent Workflow/ Finetuning/ RAG
- You Chen - Agent Workflow/ Finetuning/ Evaluation

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Layout](#project-layout)
- [Datasets](#datasets)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Fine-Tuned Models](#fine-tuned-models)
- [Running the Pipeline](#running-the-pipeline)
- [Ablation Evaluation](#ablation-evaluation)
- [Environment Variables](#environment-variables)

---

## Overview

This project implements a **Collaborative Multi-Agent** medical assistant that routes user queries through specialized agents:

| Intent | Agent | Backing Model / Data |
|---|---|---|
| Symptom diagnosis | Knowledge-Diagnosis Agent | Fine-tuned Mistral-7B on DDXPlus |
| Medication inquiry | Knowledge-Medication Agent | DrugBank GraphRAG + GPT-4o-mini |
| General medical Q&A | Knowledge-General Agent | MedQuAD VectorRAG + GPT-4o-mini |
| Medical image input | Vision Agent | Fine-tuned Qwen2-VL-7B-Instruct |

All paths produce a final patient-friendly response via a **Synthesizer** node powered by GPT-4o-mini. An **ablation framework** allows each component to be toggled independently for evaluation.

---

## System Architecture
![Project Architecture](workflow.png)

---

## Project Layout

```text
Project/
├── drugbank_graph/                  # DrugBank processing and query module
│   ├── __init__.py
│   ├── drugbank_generate.py         # Parse XML → JSONL nodes + edges
│   ├── drugbank_sqlite.py           # Build SQLite from JSONL
│   └── drugbank_query.py            # Query API (resolve, get_drug, neighbors, ddi_between)
│
├── medquad_rag/                     # MedQuAD RAG module
│   ├── __init__.py
│   ├── build_index.py               # Build FAISS index + BM25 from CSV
│   └── query_index.py               # Hybrid retrieval (vector + BM25, RRF fusion)
│
├── sft_knowledge/                   # Mistral-7B LoRA fine-tuning scripts
│   ├── train_lora.py                # LoRA training on DDXPlus SFT data
│   ├── pred_validate.py             # Run predictions on validate split
│   └── eval.py                      # Evaluate prediction accuracy
│
├── sft_qwen_vl/                     # Qwen2-VL LoRA fine-tuning scripts
│   ├── train_lora.py                # LoRA training on MM-SkinQA data
│   └── eval_lora.py                 # Evaluate vision model on skin QA
│
├── orchestration.py                 # Main LangGraph orchestration pipeline
├── evaluation.py                    # Ablation experiment evaluation framework
├── ddxplus_sft.py                   # DDXPlus → SFT JSONL converter
├── mmskin_sft.py                    # MMSkin dataset preprocessor
├── README.md
└── workflow.png
```

---

## Datasets

| Dataset | Format | Size | Purpose |
|---|---|---|---|
| **DDXPlus** | CSV splits + JSON metadata | ~1M rows | Differential diagnosis training |
| **MedQuAD** | CSV | ~47k QA pairs | General medical Q&A retrieval |
| **DrugBank** | XML | ~702 MB | Drug interactions, indications |
| **MM-SkinQA** | VQA.csv + images | ~1k+ images | Skin disease visual QA |

Place all raw datasets under `Datasets/` before running preprocessing.

---

## Installation

```bash
# Clone the repo

# Install Python dependencies
pip install torch transformers peft accelerate
pip install langchain langgraph openai
pip install sentence-transformers faiss-cpu rank-bm25
pip install pandas rouge-score nltk
pip install gradio  # for web demo only
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_openai_key_here
```

---

## Data Preprocessing

Run all commands from the **project root**. Steps must be completed in order before running the pipeline.

### 1. Build DDXPlus SFT Data

Converts DDXPlus raw CSVs into instruction-following JSONL for Mistral fine-tuning.

```bash
python ddxplus_sft.py --split train    --out processed/ddxplus_sft/train.jsonl
python ddxplus_sft.py --split validate --out processed/ddxplus_sft/validate.jsonl
python ddxplus_sft.py --split test     --out processed/ddxplus_sft/test.jsonl
```

Each record contains `instruction`, `input` (symptoms + antecedents), and `output` (JSON with `primary_diagnosis` + `differential_diagnosis`).

### 2. Build MedQuAD Vector Index

Encodes MedQuAD Q&A pairs with `all-MiniLM-L6-v2` and builds a FAISS index for dense retrieval.

```bash
python medquad_rag/build_index.py
```

Outputs: `processed/medquad/medquad_index.faiss` and `processed/medquad/medquad_corpus.jsonl`.

### 3. Build DrugBank Graph + SQLite

Parses the DrugBank XML into graph JSONL files, then loads them into an indexed SQLite database.

```bash
python drugbank_graph/drugbank_generate.py   # XML → drug_nodes.jsonl + ddi_edges.jsonl
python drugbank_graph/drugbank_sqlite.py     # JSONL → drugbank_ddi.sqlite
```

Outputs: `processed/drugbank/drug_nodes.jsonl`, `processed/drugbank/ddi_edges.jsonl`, `processed/drugbank/drugbank_ddi.sqlite`.

### 4. Preprocess MMSkin Dataset (Optional)

Merges VQA and caption CSVs, filters skin modality entries, and creates a sampled subset with images.

```bash
python mmskin_sft.py
```

Outputs: `processed/mmskin/mmskin_sft.jsonl` (full) and `processed/mmskin/mmskin_sft_small.jsonl` (1280 samples).

---

## Fine-Tuned Models

The pipeline requires two LoRA-adapted models. Place adapter directories at the project root (or override via environment variables):

| Model | Adapter Directory | Trained On |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistral_7b/` | DDXPlus SFT JSONL |
| `Qwen/Qwen2-VL-7B-Instruct` | `qwen_vl_lora/` | MM-SkinQA JSONL |

Both models are loaded with PEFT and swapped between GPU and CPU dynamically to manage VRAM.

---

## Running the Pipeline

```bash
python orchestration.py
```

The script runs a built-in smoke test with a sample symptom query. To integrate into your own code:

```python
from orchestration import run_pipeline

result = run_pipeline(
    user_text="I have a headache and fever for two days.",
    user_image=None,          # optional: path to a medical image
    use_memory=False          # set True to enable conversation checkpointing
)
print(result["final_response"])
```

### Ablation Flags

Each agent can be toggled independently via `ablation_flags`:

```python
result = run_pipeline(
    user_text="Does aspirin interact with warfarin?",
    ablation_flags={
        "use_vision":              True,   # Vision Agent
        "use_diagnosis_agent":     True,   # Fine-tuned Mistral-7B
        "use_medication_graphrag": False,  # DrugBank GraphRAG (disabled)
        "use_general_vectorrag":   True,   # MedQuAD VectorRAG
    }
)
```

---

## Ablation Evaluation

Runs systematic ablation experiments across all four components and reports metrics.

```bash
python evaluation.py
```

| Component | Evaluation Dataset | Metrics |
|---|---|---|
| Knowledge-Diagnosis Agent | DDXPlus test set | Primary accuracy, Recall@K |
| Vision Agent | MM-SkinQA | ROUGE-L, BLEU |
| Knowledge-Medication Agent (GraphRAG) | DrugBank DDI pairs | ROUGE-L, BLEU |
| Knowledge-General Agent (VectorRAG) | MedQuAD | ROUGE-L, BLEU |

Results are saved as JSON files under `processed/*/ab_w_*/` (with component) and `processed/*/ab_wo_*/` (without component).

---

## Environment Variables

`orchestration.py` reads the following optional environment overrides:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for GPT-4o-mini |
| `DRUGBANK_DB_PATH` | `processed/drugbank/drugbank_ddi.sqlite` | Path to DrugBank SQLite DB |
| `DIAGNOSIS_ADAPTER_DIR` | `mistral_7b/` | LoRA adapter for Mistral-7B diagnosis model |
| `VISION_ADAPTER_DIR` | `qwen_vl_lora/` | LoRA adapter for Qwen2-VL vision model |
