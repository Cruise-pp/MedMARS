# Evaluation script for the medical agent with ablation experiments
import json
import random
import sqlite3
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from orchestration import run_turn

# Set random seed for reproducibility
random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
PROCESSED_DIR = PROJECT_ROOT / "processed"


# ==========================================
# Helper Functions
# ==========================================
def get_base_flags() -> Dict[str, bool]:
    """Returns the default ablation flags."""
    return {
        "use_vision": True,
        "use_diagnosis_agent": True,
        "use_medication_graphrag": True,
        "use_general_vectorrag": True
    }

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Helper to load JSONL files."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """Helper to save data to a JSON file."""
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ==========================================
# Evaluation Functions
# ==========================================
def eval_knowledge_agent(
    input_data: List[Dict[str, Any]], 
    num_data: int, 
    k: int, 
    output_path: str, 
    is_ablation: bool = False
) -> None:
    """
    Evaluates the knowledge agent using accuracy and Hit@K.
    """
    flags = get_base_flags()
    if is_ablation:
        flags["use_diagnosis_agent"] = False
    
    input_data = random.sample(input_data, min(num_data, len(input_data)))
    results = []
    acc1, hitk = 0, 0

    for d in tqdm(input_data, desc="Evaluating Knowledge Agent"):
        user_text = f"{d['instruction']}\n{d['input']}"
        answer = json.loads(d["output"])
        gt_primary = (answer.get("primary_diagnosis") or "").strip().replace("\"", "")
        
        model_output = run_turn(user_text=user_text, ablation_flags=flags, use_memory=False)
        d["model_output"] = model_output
        results.append(d)

        # Primary accuracy
        if gt_primary.lower() in model_output.lower():
            acc1 += 1

        # Hit@k
        topk = answer.get("differential_diagnosis", [])[:k]
        for ele in topk:
            if ele["label"].replace("\"", "").lower().strip() in model_output.lower():
                hitk += 1
                break

    acc1 /= len(input_data)
    hitk /= len(input_data)

    print(f"\nKnowledge Agent Results -> Primary accuracy: {acc1:.3f} | Recall@{k}: {hitk:.3f}")
    save_json(results, output_path)


def eval_qa_agent(
    input_data: List[Dict[str, Any]], 
    num_data: int, 
    output_path: str, 
    ablation_target: Optional[str] = None,
    image_dir: Optional[str] = None,
    sample_from_end: bool = False
) -> None:
    """
    Unified evaluation function for Vision, GraphRAG, and VectorRAG agents.
    Calculates ROUGE-L and BLEU scores.
    """
    flags = get_base_flags()
    if ablation_target:
        flags[ablation_target] = False
        
    num_data = min(num_data, len(input_data))
    
    # Vision agent originally sampled from the end; others use random sample
    if sample_from_end:
        eval_data = input_data[-num_data:]
    else:
        eval_data = random.sample(input_data, num_data)

    results = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    total_rouge, total_bleu = 0.0, 0.0 

    for d in tqdm(eval_data, desc=f"Evaluating QA Agent (Ablating: {ablation_target})"):
        question = d["question"]
        answer = d["answer"]
        
        # Attach image path if evaluating vision
        image_path = str(Path(image_dir) / d["image"]) if image_dir and "image" in d else None
        
        model_output = run_turn(
            user_text=question, 
            user_image=image_path, 
            ablation_flags=flags, 
            use_memory=False
        )
        
        # Compute ROUGE-L
        rouge_l = scorer.score(answer, model_output)['rougeL'].fmeasure
        
        # Compute BLEU
        bleu_score = sentence_bleu([answer.split()], model_output.split(), smoothing_function=smoothing)
        
        d.update({
            "model_output": model_output,
            "rouge_l": rouge_l,
            "bleu": bleu_score
        })
        
        total_bleu += bleu_score
        total_rouge += rouge_l
        results.append(d)
    
    avg_rouge = total_rouge / num_data
    avg_bleu = total_bleu / num_data
    
    print(f"\nQA Results -> Avg ROUGE-L: {avg_rouge:.4f} | Avg BLEU: {avg_bleu:.4f}")
    save_json(results, output_path)


# ==========================================
# Data Construction from drugbank database
# ==========================================
def construct_graphrag_data(db_path: str, output_path: str) -> None:
    """Constructs the evaluation dataset from the DrugBank SQLite database."""
    dataset = []

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT src_name, dst_name, description FROM ddi_edges")
            
            for src_name, dst_name, description in cursor.fetchall():
                if not all([src_name, dst_name, description]):
                    continue
                
                dataset.append({
                    "question": f"What is the relation between {src_name} and {dst_name}?",
                    "answer": description
                })
                
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return

    save_json(dataset, output_path)
    print(f"Successfully generated {len(dataset)} GraphRAG entries at '{output_path}'.")


# ==========================================
# Main Execution Pipeline
# ==========================================

if __name__ == "__main__":
    
    NUM_DATA_SAMPLES = 100
    
    # 1. Knowledge Agent Evaluation
    print("\n--- Running Knowledge Agent Evaluation ---")
    # Sampled test data from ddxplus test dataset
    ddx_data = load_jsonl(str(PROCESSED_DIR / "ddxplus_sft/test.jsonl"))
    
    print("Evaluating with Knowledge Agent (Baseline)")
    eval_knowledge_agent(
        input_data=ddx_data, num_data=NUM_DATA_SAMPLES, k=5, 
        output_path=str(PROCESSED_DIR / "ddxplus_sft/ab_w_knowledge.json"), 
        is_ablation=False
    )
    print("Evaluating without Knowledge Agent (Ablated)")
    eval_knowledge_agent(
        input_data=ddx_data, num_data=NUM_DATA_SAMPLES, k=5, 
        output_path=str(PROCESSED_DIR / "ddxplus_sft/ab_wo_knowledge.json"), 
        is_ablation=True
    )

    # 2. Vision Agent Evaluation
    print("\n--- Running Vision Agent Evaluation ---")
    mmskin_data = load_jsonl(str(PROCESSED_DIR / "mmskin/mmskin_sft_small.jsonl"))
    
    print("Evaluating with Vision Agent (Baseline)")
    eval_qa_agent(
        input_data=mmskin_data, 
        num_data=NUM_DATA_SAMPLES, 
        output_path=str(PROCESSED_DIR / "mmskin/ab_w_vision.json"), 
        ablation_target=None, # Baseline
        image_dir=str(PROCESSED_DIR / "mmskin/MM-SkinQA-small"),
        sample_from_end=True
    )
    print("Evaluating without Vision Agent (Ablated)")
    eval_qa_agent(
        input_data=mmskin_data, 
        num_data=NUM_DATA_SAMPLES, 
        output_path=str(PROCESSED_DIR / "mmskin/ab_wo_vision.json"), 
        ablation_target="use_vision", # Ablated
        image_dir=str(PROCESSED_DIR / "mmskin/MM-SkinQA-small"),
        sample_from_end=True
    )

    # 3. GraphRAG Evaluation
    print("\n--- Running GraphRAG Evaluation ---")
    with open(PROCESSED_DIR / "drugbank/drugbank_q.json", "r") as f:
        drugbank_data = json.load(f)
        
    print("Evaluating with GraphRAG (Baseline)")
    eval_qa_agent(
        input_data=drugbank_data, 
        num_data=NUM_DATA_SAMPLES, 
        output_path=str(PROCESSED_DIR / "drugbank/ab_w_graphrag.json"), 
        ablation_target=None # Baseline
    )
    print("Evaluating without GraphRAG (Ablated)")
    eval_qa_agent(
        input_data=drugbank_data, 
        num_data=NUM_DATA_SAMPLES, 
        output_path=str(PROCESSED_DIR / "drugbank/ab_wo_graphrag.json"), 
        ablation_target="use_medication_graphrag" # Ablated
    )

    # 4. MedQA (VectorRAG) Evaluation
    print("\n--- Running VectorRAG Evaluation ---")
    df = pd.read_csv(DATASETS_DIR / "MedQuAD.csv")
    medquad_data = [{"question": row["question"], "answer": row["answer"]} for _, row in df.iterrows()]
    
    print("Evaluating with VectorRAG (Baseline)")
    eval_qa_agent(
        input_data=medquad_data, 
        num_data=NUM_DATA_SAMPLES, 
        output_path=str(PROCESSED_DIR / "medquad/ab_w_vectorrag.json"), 
        ablation_target=None # Baseline
    )
    print("Evaluating without VectorRAG (Ablated)")
    eval_qa_agent(
        input_data=medquad_data, 
        num_data=NUM_DATA_SAMPLES, 
        output_path=str(PROCESSED_DIR / "medquad/ab_wo_vectorrag.json"), 
        ablation_target="use_general_vectorrag" # Ablated
    )
