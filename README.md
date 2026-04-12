# MedMARS

**Multi-Agent Retrieval System for Medicine**

A multi-agent medical AI system built on **LangGraph** that combines fine-tuned language models, vision models, and retrieval-augmented generation (RAG) to assist with medical diagnosis, medication interaction queries, and general medical Q&A.

---

## Architecture

![System Architecture](workflow.png)

| Intent | Agent | Backing Model / Data |
|---|---|---|
| Symptom diagnosis | Knowledge-Diagnosis | Fine-tuned Mistral-7B on DDXPlus |
| Medication inquiry | Knowledge-Medication | DrugBank GraphRAG + GPT-4o-mini |
| General medical Q&A | Knowledge-General | MedQuAD VectorRAG + GPT-4o-mini |
| Medical image input | Vision Agent | Fine-tuned Qwen2-VL-7B-Instruct |

All paths produce a final patient-friendly response via a **Synthesizer** node (GPT-4o-mini) with rolling conversation summary.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Cruise-pp/MedMARS.git
cd MedMARS

# 2. Install dependencies
pip install torch transformers peft accelerate
pip install langchain langgraph openai
pip install sentence-transformers faiss-cpu rank-bm25
pip install pandas rouge-score nltk rich prompt_toolkit

# 3. Install MedMARS CLI
pip install -e .

# 4. Set your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 5. Prepare data (see Data Preparation section below)

# 6. Launch
medmars
```

---

## Usage

### CLI

Run `medmars` from any terminal after installation.

```
Commands:
  /image <path> <question>   Analyze a medical image
  /summary                   Show conversation summary
  /new                       Start a new conversation
  /help                      Show this help
  /quit                      Exit
```

### Python API

```python
from orchestration import run_turn

response = run_turn(
    user_text="I have a headache and fever for two days.",
    user_image=None,       # optional: path to a medical image
    thread_id="session_1", # conversation thread ID
    ablation_flags={       # toggle components on/off
        "use_vision": True,
        "use_diagnosis_agent": True,
        "use_medication_graphrag": True,
        "use_general_vectorrag": True,
    },
)
print(response)
```

### MCP Server

`mcp_medical_server.py` exposes MedQuAD search, DrugBank drug lookup, and drug interaction checks as [MCP](https://modelcontextprotocol.io/) tools. Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "medical-knowledge": {
      "command": "python",
      "args": ["mcp_medical_server.py"]
    }
  }
}
```

---

## Project Layout

```text
MedMARS/
├── cli.py                           # CLI entry point (medmars command)
├── pyproject.toml                   # Package config for pip install -e .
├── orchestration.py                 # Main LangGraph orchestration pipeline
├── mcp_medical_server.py            # MCP server (MedQuAD + DrugBank tools)
├── evaluation.py                    # Ablation evaluation framework
├── eval_retrieval.py                # RAG retrieval metrics (Recall, MRR, NDCG)
│
├── drugbank_graph/                  # DrugBank processing and query module
│   ├── drugbank_generate.py         #   Parse XML → JSONL nodes + edges
│   ├── drugbank_sqlite.py           #   Build SQLite from JSONL
│   └── drugbank_query.py            #   Query API (resolve, get_drug, neighbors, ddi)
│
├── medquad_rag/                     # MedQuAD RAG module
│   ├── build_index.py               #   Build FAISS index + BM25 from CSV
│   └── query_index.py               #   Hybrid retrieval (vector + BM25, RRF fusion)
│
├── sft_knowledge/                   # Mistral-7B LoRA fine-tuning
│   ├── train_lora.py                #   LoRA training on DDXPlus SFT data
│   ├── pred_validate.py             #   Run predictions on validate split
│   └── eval.py                      #   Evaluate prediction accuracy
│
├── sft_qwen_vl/                     # Qwen2-VL LoRA fine-tuning
│   ├── train_lora.py                #   LoRA training on MM-SkinQA data
│   └── eval_lora.py                 #   Evaluate vision model on skin QA
│
├── ddxplus_sft.py                   # DDXPlus → SFT JSONL converter
├── mmskin_sft.py                    # MMSkin dataset preprocessor
└── workflow.png                     # Architecture diagram
```

---

## Data Preparation

Place raw datasets under `Datasets/`, then run preprocessing in order:

### 1. DDXPlus SFT Data

```bash
python ddxplus_sft.py --split train    --out processed/ddxplus_sft/train.jsonl
python ddxplus_sft.py --split validate --out processed/ddxplus_sft/validate.jsonl
python ddxplus_sft.py --split test     --out processed/ddxplus_sft/test.jsonl
```

### 2. MedQuAD Vector Index

```bash
python medquad_rag/build_index.py
```

### 3. DrugBank Graph + SQLite

```bash
python drugbank_graph/drugbank_generate.py
python drugbank_graph/drugbank_sqlite.py
```

### 4. MMSkin Dataset (optional, for vision model training)

```bash
python mmskin_sft.py
```

### Datasets

| Dataset | Format | Purpose |
|---|---|---|
| **DDXPlus** | CSV splits + JSON metadata | Differential diagnosis training |
| **MedQuAD** | CSV (~47k QA pairs) | General medical Q&A retrieval |
| **DrugBank** | XML (~702 MB) | Drug interactions, indications |
| **MM-SkinQA** | VQA.csv + images | Skin disease visual QA |

---

## Models

Two LoRA-adapted models are required. Place adapter directories at the project root or override via environment variables.

| Base Model | Adapter Directory | Trained On |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistral7b_lora/` | DDXPlus SFT |
| `Qwen/Qwen2-VL-7B-Instruct` | `qwen_vl_lora/` | MM-SkinQA |

Both models are loaded with PEFT and swapped between GPU/CPU dynamically to manage VRAM.

---

## Evaluation

### Ablation Experiments

Toggle each component independently to measure its contribution:

```bash
python evaluation.py
```

| Component | Dataset | Metrics |
|---|---|---|
| Knowledge-Diagnosis Agent | DDXPlus test set | Primary accuracy, Recall@K |
| Vision Agent | MM-SkinQA | ROUGE-L, BLEU |
| Knowledge-Medication Agent | DrugBank DDI pairs | ROUGE-L, BLEU |
| Knowledge-General Agent | MedQuAD | ROUGE-L, BLEU |

### Retrieval Quality

Evaluate RAG retrieval performance (MedQuAD VectorRAG and DrugBank GraphRAG):

```bash
python eval_retrieval.py
```

Reports Recall@K, MRR@K, and NDCG@K for both retrieval backends.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for GPT-4o-mini |
| `DRUGBANK_DB_PATH` | `processed/drugbank/drugbank_ddi.sqlite` | DrugBank SQLite database |
| `DIAGNOSIS_ADAPTER_DIR` | `mistral7b_lora/` | LoRA adapter for Mistral-7B |
| `VISION_ADAPTER_DIR` | `qwen_vl_lora/` | LoRA adapter for Qwen2-VL |

All variables can be set in a `.env` file at the project root.

---

## Team

- **Yufan Shi** — Data Preprocessing / Agent Workflow / Fine-tuning / RAG
- **You Chen** — Agent Workflow / Fine-tuning / Evaluation
