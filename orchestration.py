"""
Graph-VMA Orchestration Pipeline (Ablation-Ready)
=================================================
Multi-agent medical assistant built on LangGraph.

Architecture:
  Orchestrator (DeepSeek V3 intent classification)
    ├─ symptom_diagnosis  → KnowledgeAgent-Diagnosis (fine-tuned LLM / Ablation)
    ├─ medication_inquiry → KnowledgeAgent-Medication (GraphRAG + LLM / Ablation)
    ├─ general_qa         → KnowledgeAgent-General    (RAG + LLM / Ablation)
    └─ has image?         → VisionAgent (Ablation)
  All knowledge paths    → Synthesizer (DeepSeek V3) → END
"""

import json
import operator
import torch
import gc
import os
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    HumanMessage, SystemMessage, BaseMessage, AIMessage,
)
from medquad_rag import query_index as mq
from drugbank_graph import drugbank_query as dq
from typing import TypedDict, Annotated, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from openai import OpenAI
from PIL import Image

# ================================================================
# Config
# ================================================================
OPENAI_API_KEY = "xxx"                          # TODO: 换成你的 key
OPENAI_MODEL = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).resolve().parent

# DrugBank SQLite path (used by medication branch)
DRUGBANK_DB_PATH = os.getenv(
    "DRUGBANK_DB_PATH",
    str(PROJECT_ROOT / "processed/drugbank/drugbank_ddi.sqlite"),
)

# Config
DIAGNOSIS_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DIAGNOSIS_ADAPTER_DIR = os.getenv("DIAGNOSIS_ADAPTER_DIR", str(PROJECT_ROOT / "mistral_7b"))
VISION_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
VISION_ADAPTER_DIR = os.getenv("VISION_ADAPTER_DIR", str(PROJECT_ROOT / "qwen_vl_lora"))

# ================================================================
# 1. State
# ================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_text: str
    user_image: Optional[str]           
    intent: Optional[str]               
    clinical_evidence: Optional[str]    
    retrieved_context: Optional[str]    
    diagnosis_output: Optional[str]     
    final_response: str
    ablation_flags: dict                # NEW: Dictionary to control ablation


# ================================================================
# 2. LLM helpers
# ================================================================
def call_openai(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call OpenAI's gpt-4o-mini model."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


# ================================================================
# 3. Orchestrator  — intent classification via gpt
# ================================================================
_INTENT_SYSTEM = """\
You are a medical triage router. Classify the user query into EXACTLY ONE label:

  symptom_diagnosis   – user describes symptoms / complaints / asks for diagnosis
  medication_inquiry  – user asks about drugs, drug interactions, side effects, dosage
  general_qa          – general medical knowledge question

Reply with the label ONLY (one line, no quotes, no explanation)."""

_VALID_INTENTS = {"symptom_diagnosis", "medication_inquiry", "general_qa"}

def orchestrator(state: AgentState) -> dict:
    user_text = state.get("user_text", "")
    if not user_text.strip():
        return {"intent": "general_qa"}

    raw = call_openai(_INTENT_SYSTEM, user_text)
    intent = raw.strip().strip('"').strip("'").lower().replace(" ", "_")

    if intent not in _VALID_INTENTS:
        intent = "general_qa"

    print(f"[Orchestrator] intent = {intent}")
    return {
        "intent": intent,
        "messages": [SystemMessage(content=f"[Orchestrator] intent={intent}")],
    }


# ================================================================
# Memory Manager
# ================================================================
_active_gpu_model = None

def _manage_vram(target_model_name: str):
    """Swaps models between CPU RAM and GPU VRAM to optimize inference time."""
    global _active_gpu_model, _vision_model, _diag_model
    
    if _active_gpu_model == target_model_name:
        return 

    print(f"\n[Memory Manager] VRAM swap: Moving {target_model_name} to GPU...")

    # Offload inactive model to CPU
    if target_model_name == "diagnosis" and _vision_model is not None:
        _vision_model.to("cpu")
    elif target_model_name == "vision" and _diag_model is not None:
        _diag_model.to("cpu")

    # Flush VRAM cache completely
    gc.collect()
    torch.cuda.empty_cache()

    # Load target model to GPU
    try:
        if target_model_name == "diagnosis" and _diag_model is not None:
            _diag_model.to("cuda")
        elif target_model_name == "vision" and _vision_model is not None:
            _vision_model.to("cuda")
            
        _active_gpu_model = target_model_name
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"[Memory Manager] FATAL: CUDA Out of Memory Error while loading {target_model_name}. {e}")
        raise e


# ================================================================
# 4. Vision Agent  (Qwen2-VL-7B-Instruct)
# ================================================================
_vision_model = None
_vision_processor = None

def _ensure_vision_model_loaded():
    global _vision_model, _vision_processor
    if _vision_model is not None and _vision_processor is not None:
        return

    print("[Vision Agent] Loading base model + LoRA adapter to CPU...")
    
    _vision_processor = AutoProcessor.from_pretrained(VISION_BASE_MODEL)
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        VISION_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"":"cpu"},
        attn_implementation="sdpa",
    )
    _vision_model = PeftModel.from_pretrained(base_model, VISION_ADAPTER_DIR)
    _vision_model = _vision_model.merge_and_unload()
    _vision_model.eval()
    print("[Vision Agent] Model loaded to CPU.")

def vision_agent(state: AgentState) -> dict:
    _ensure_vision_model_loaded()
    _manage_vram("vision")
    
    image_path = state.get("user_image", "")
    image = Image.open(image_path).convert("RGB")
    print(f"[Vision Agent] Processing image: {image_path}")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": state.get("user_text", "")}
            ]
        }
    ]

    prompt = _vision_processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _vision_processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(_vision_model.device)

    try:
        with torch.no_grad():
            generated_ids = _vision_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[Vision Agent] CUDA Out of Memory Error during generation: {e}")
        raise e

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
        
    clinical_evidence = _vision_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0].strip()

    print(f"[Vision Agent] Clinical evidence: {clinical_evidence}")
    return {"clinical_evidence": clinical_evidence}


# ================================================================
# 5a. Knowledge Agent — Diagnosis  (Mistral-7B)
# ================================================================
_diag_model = None
_diag_tokenizer = None

def _ensure_diagnosis_model_loaded():
    # ... [Keep your existing _ensure_diagnosis_model_loaded code here] ...
    global _diag_model, _diag_tokenizer
    if _diag_model is not None:
        return

    print("[Knowledge-Diagnosis] Loading base model + LoRA adapter to CPU...")
    _diag_tokenizer = AutoTokenizer.from_pretrained(
        DIAGNOSIS_BASE_MODEL, trust_remote_code=True
    )
    if _diag_tokenizer.pad_token is None:
        _diag_tokenizer.pad_token = _diag_tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        DIAGNOSIS_BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"":"cpu"},
    )

    if hasattr(base, "_no_split_modules") and base._no_split_modules is not None:
        clean_modules = []
        for mod in base._no_split_modules:
            if isinstance(mod, (list, tuple, set)):
                clean_modules.extend(list(mod))
            else:
                clean_modules.append(mod)
        base._no_split_modules = list(set(clean_modules))

    _diag_model = PeftModel.from_pretrained(base, DIAGNOSIS_ADAPTER_DIR)
    _diag_model.eval()
    print("[Knowledge-Diagnosis] Model loaded to CPU.")

_SYMPTOM_EXTRACT_SYSTEM = """\
Extract structured medical information from the patient's description.
Return a JSON object with these fields:
{
  "age": <number or null>,
  "sex": "<M or F or null>",
  "symptoms": ["symptom1", "symptom2", ...],
  "antecedents": ["risk factor1", ...]
}

Rules:
- "symptoms" = current complaints and findings
- "antecedents" = pre-existing conditions, lifestyle risk factors, medical history
- If age or sex is not mentioned, use null
- Keep symptom descriptions concise
- Reply with ONLY the JSON object, nothing else"""

def _extract_symptoms(user_text: str, clinical_evidence: str = "") -> dict:
    prompt = user_text
    if clinical_evidence:
        prompt += f"\n\nAdditional clinical evidence from imaging:\n{clinical_evidence}"

    raw = call_openai(_SYMPTOM_EXTRACT_SYSTEM, prompt)
    cleaned = raw.strip().strip("`").strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return {"age": None, "sex": None, "symptoms": [], "antecedents": []}


_SFT_INSTRUCTION = (
    "You are an expert in medical diagnostic reasoning."
    "Analyze the patient's demographics, initial complaint, and symptoms/antecedents/risk factors."
    "Provide a structured assessment in valid JSON format with two fields: "
    '"primary_diagnosis" (the single most likely diagnosis label) and '
    '"differential_diagnosis" (a list of top candidate diseases with their probabilities).'
    "Do not output any conversational text, only the JSON object."
)

_SFT_PROMPT_TMPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input_text}\n\n"
    "### Output:\n"
)

def _format_sft_prompt(extracted: dict) -> str:
    age = extracted.get("age")
    sex = extracted.get("sex")
    symptoms = extracted.get("symptoms", [])
    antecedents = extracted.get("antecedents", [])

    age_str = str(age) if age is not None else "unknown"
    sex_str = str(sex) if sex is not None else "unknown"
    header = f"Patient: age={age_str}, sex={sex_str}"

    parts = [header]
    if symptoms:
        symptom_lines = [f"- {s} YES" for s in symptoms]
        parts.append("Symptoms & Current findings:\n" + "\n".join(symptom_lines))
    if antecedents:
        ante_lines = [f"- {a} YES" for a in antecedents]
        parts.append("Antecedents & Risk factors:\n" + "\n".join(ante_lines))

    input_text = "\n\n".join(parts)

    return _SFT_PROMPT_TMPL.format(
        instruction = _SFT_INSTRUCTION,
        input_text = input_text,
    )

def _run_diagnosis_inference(prompt: str, max_new_tokens: int = 512) -> str:
    assert _diag_model is not None and _diag_tokenizer is not None

    inputs = _diag_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )
    inputs = {k: v.to(_diag_model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            out_ids = _diag_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_diag_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[Knowledge-Diagnosis] CUDA Out of Memory Error during generation: {e}")
        raise e

    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return _diag_tokenizer.decode(gen_ids, skip_special_tokens=True)

def _parse_diagnosis_json(raw: str) -> dict:
    t = (raw or "").strip()
    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(t[i:j+1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"primary_diagnosis": "unknown", "differential_diagnosis": []}

def knowledge_diagnosis(state: AgentState) -> dict:
    flags = state.get("ablation_flags", {})
    if not flags.get("use_diagnosis_agent", True):
        print("[Knowledge-Diagnosis] Ablation: Skipping fine-tuned diagnosis model.")
        return {
            "diagnosis_output": json.dumps({
                "primary_diagnosis": "Ablated - Model Disabled", 
                "differential_diagnosis": []
            })
        }

    user_text = state.get("user_text", "")
    clinical_evidence = state.get("clinical_evidence", "")
    print(f"[Knowledge-Diagnosis] query='{user_text[:80]}'")

    extracted = _extract_symptoms(user_text, clinical_evidence)
    prompt = _format_sft_prompt(extracted)

    _ensure_diagnosis_model_loaded()
    _manage_vram("diagnosis")
    
    raw_output = _run_diagnosis_inference(prompt)
    diagnosis = _parse_diagnosis_json(raw_output)
    diagnosis_output = json.dumps(diagnosis, ensure_ascii=False)

    return {"diagnosis_output": diagnosis_output}


# ================================================================
# 5b. Knowledge Agent — Medication  (GraphRAG + gpt)
# ================================================================

_DRUG_EXTRACT_SYSTEM = """\
Extract all drug / medication names from the user query.
Return a JSON array of strings, e.g. ["ibuprofen", "aspirin"].
If no drugs are mentioned, return [].
Reply with ONLY the JSON array, nothing else."""

def _extract_drug_names(user_text: str) -> list[str]:
    raw = call_openai(_DRUG_EXTRACT_SYSTEM, user_text)
    cleaned = raw.strip().strip("`").strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        names = json.loads(cleaned)
        if isinstance(names, list):
            return [str(n).strip() for n in names if str(n).strip()]
    except json.JSONDecodeError:
        pass
    return []

def knowledge_medication(state: AgentState) -> dict:
    user_text = state.get("user_text", "")
    drug_names = _extract_drug_names(user_text)

    flags = state.get("ablation_flags", {})
    if not flags.get("use_medication_graphrag", True):
        print("[Knowledge-Medication] Ablation: Skipping GraphRAG retrieval.")
        return {
            "retrieved_context": "GraphRAG Ablated",
            "diagnosis_output": json.dumps({"drugs_identified": drug_names, "interactions": []}),
        }

    print(f"[Knowledge-Medication] query='{user_text[:80]}'")
    print(f"[Knowledge-Medication] extracted drugs: {drug_names}")

    if not drug_names:
        return {
            "retrieved_context": "",
            "diagnosis_output": json.dumps({
                "drugs_identified": [],
                "interactions": [],
                "safety_assessment": "No drug names detected in the query.",
            }),
        }

    resolved: list[dict] = []          
    failed:   list[str]  = []          

    for name in drug_names:
        r = dq.resolve(name, db_path=DRUGBANK_DB_PATH)
        if r["status"] == "not_found":
            failed.append(name)
        else:
            top = r["candidates"][0]
            drug_info = dq.get_drug(top["drug_id"], db_path=DRUGBANK_DB_PATH)
            resolved.append({
                "query":       name,
                "drug_id":     top["drug_id"],
                "name":        top["name"],
                "indication":  (drug_info.get("drug") or {}).get("indication", ""),
                "description": (drug_info.get("drug") or {}).get("description", ""),
            })

    interactions: list[dict] = []

    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            a = resolved[i]
            b = resolved[j]
            ddi = dq.ddi_between(
                a["drug_id"], b["drug_id"],
                db_path=DRUGBANK_DB_PATH,
            )
            if ddi["status"] == "found":
                interactions.append({
                    "drug_a": a["name"],
                    "drug_b": b["name"],
                    "evidence": ddi["evidence"],
                })

    ctx_parts: list[str] = []

    for d in resolved:
        lines = f"Drug: {d['name']} ({d['drug_id']})"
        if d["indication"]:
            lines += f"\n  Indication: {d['indication'][:300]}"
        ctx_parts.append(lines)

    for inter in interactions:
        header = f"Interaction: {inter['drug_a']} ↔ {inter['drug_b']}"
        evidence_text = "\n  ".join(inter["evidence"][:3])  
        ctx_parts.append(f"{header}\n  {evidence_text}")

    if failed:
        ctx_parts.append(f"Unresolved drugs: {', '.join(failed)}")

    retrieved_context = "\n\n".join(ctx_parts)

    diagnosis_output = json.dumps({
        "drugs_identified": [
            {"name": d["name"], "drug_id": d["drug_id"]} for d in resolved
        ],
        "unresolved": failed,
        "interactions": [
            {
                "drug_a": inter["drug_a"],
                "drug_b": inter["drug_b"],
                "evidence_count": len(inter["evidence"]),
                "evidence_preview": inter["evidence"][0][:200] if inter["evidence"] else "",
            }
            for inter in interactions
        ],
        "has_interactions": len(interactions) > 0,
    }, ensure_ascii=False)

    return {
        "retrieved_context": retrieved_context,
        "diagnosis_output":  diagnosis_output,
    }


# ================================================================
# 5c. Knowledge Agent — General QA  (MedQuAD RAG + gpt)
# ================================================================
_GENERAL_QA_SYSTEM = """\
You are a medical knowledge assistant. Answer the patient's question accurately
based on the provided reference materials.

Rules:
- Base your answer on the provided context; do not fabricate information
- If the context does not fully answer the question, say so honestly
- Use clear, professional language
- Cite specific details from the context when relevant"""

def knowledge_general(state: AgentState) -> dict:
    user_text = state.get("user_text", "")
    flags = state.get("ablation_flags", {})
    
    if not flags.get("use_general_vectorrag", True):
        print("[Knowledge-General] Ablation: Skipping VectorRAG, answering via raw LLM.")
        answer = call_openai(_GENERAL_QA_SYSTEM, f"Patient question: {user_text}", max_tokens=1024)
        return {
            "retrieved_context": "VectorRAG Ablated",
            "diagnosis_output":  json.dumps({"answer": answer}, ensure_ascii=False)
        }

    from medquad_rag import query_index as mq

    print(f"[Knowledge-General] query='{user_text[:80]}'")
    results = mq.search(user_text, top_k=5)

    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Reference {i+1}]\n"
            f"Q: {r['question']}\n"
            f"A: {r['answer']}"
        )
    retrieved_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

    if retrieved_context:
        user_prompt = (
            f"Reference materials:\n{retrieved_context}\n\n"
            f"Patient question: {user_text}\n\n"
            f"Provide a comprehensive answer based on the references above:"
        )
        answer = call_openai(_GENERAL_QA_SYSTEM, user_prompt, max_tokens=1024)
    else:
        answer = ("I could not find relevant medical references for your question. "
                  "Please consult a healthcare professional for accurate guidance.")

    diagnosis_output = json.dumps({"answer": answer}, ensure_ascii=False)

    return {
        "retrieved_context": retrieved_context,
        "diagnosis_output":  diagnosis_output,
    }


# ================================================================
# 6. Synthesizer  (gpt)
# ================================================================
_SYNTH_SYSTEM = """\
You are a compassionate medical assistant. Convert the clinical reasoning below
into a clear, empathetic, and safety-aware response for the patient.

Rules:
- Use simple language the patient can understand
- Always recommend seeing a healthcare professional for serious concerns
- Be empathetic but factual; do NOT fabricate information"""
# You can add a more rules. E.g.
# - Be brief and concise, with only 1 sentence.

def synthesizer(state: AgentState) -> dict:
    diagnosis_output  = state.get("diagnosis_output",  "")
    clinical_evidence = state.get("clinical_evidence",  "")
    retrieved_context = state.get("retrieved_context",  "")
    user_text         = state.get("user_text", "")

    parts = []
    if clinical_evidence:
        parts.append(f"[Visual Findings]\n{clinical_evidence}")
    if retrieved_context:
        parts.append(f"[Retrieved Knowledge]\n{retrieved_context}")
    if diagnosis_output:
        parts.append(f"[Clinical Reasoning]\n{diagnosis_output}")

    context = "\n\n".join(parts) if parts else "(no context)"

    user_prompt = (
        f"Patient query: {user_text}\n\n"
        f"Clinical context:\n{context}\n\n"
        f"Generate a patient-friendly response:"
    )

    final_response = call_openai(_SYNTH_SYSTEM, user_prompt)

    print(f"[Synthesizer] response length = {len(final_response)}")
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
    }


# ================================================================
# 7. Routing functions
# ================================================================
def route_after_orchestrator(state: AgentState) -> str:
    flags = state.get("ablation_flags", {})
    
    # Check if Vision is disabled via ablation flags
    if state.get("user_image") and flags.get("use_vision", True):
        return "vision_agent"

    intent = state.get("intent", "general_qa")
    _INTENT_TO_NODE = {
        "symptom_diagnosis":  "knowledge_diagnosis",
        "medication_inquiry": "knowledge_medication",
        "general_qa":         "knowledge_general",
    }
    return _INTENT_TO_NODE[intent] if intent in _INTENT_TO_NODE else "knowledge_general"

# ================================================================
# 8. Build LangGraph
# ================================================================
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator",         orchestrator)
workflow.add_node("vision_agent",         vision_agent)
workflow.add_node("knowledge_diagnosis",  knowledge_diagnosis)
workflow.add_node("knowledge_medication", knowledge_medication)
workflow.add_node("knowledge_general",    knowledge_general)
workflow.add_node("synthesizer",          synthesizer)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {
        "vision_agent":         "vision_agent",
        "knowledge_diagnosis":  "knowledge_diagnosis",
        "knowledge_medication": "knowledge_medication",
        "knowledge_general":    "knowledge_general",
    },
)

workflow.add_edge("vision_agent", "synthesizer")
workflow.add_edge("knowledge_diagnosis",  "synthesizer")
workflow.add_edge("knowledge_medication", "synthesizer")
workflow.add_edge("knowledge_general",    "synthesizer")
workflow.add_edge("synthesizer", END)

# NEW: Compile two distinct versions of the app
memory_checkpointer = MemorySaver()
app_with_memory = workflow.compile(checkpointer=memory_checkpointer)
app_without_memory = workflow.compile() # Completely stateless


# ================================================================
# 9. Convenience runner (for testing / ablations)
# ================================================================
def run_turn(
    user_text: str,
    user_image: str | None = None,
    thread_id: str = "default",
    ablation_flags: dict = None,
    use_memory: bool = True,             # NEW: Toggle memory on/off
) -> str:
    # Default to using all agents if no flags are passed
    if ablation_flags is None:
        ablation_flags = {
            "use_vision": True,
            "use_diagnosis_agent": True,
            "use_medication_graphrag": True,
            "use_general_vectorrag": True
        }

    inputs: AgentState = {                          # type: ignore[typeddict-item]
        "messages":           [HumanMessage(content=user_text)],
        "user_text":          user_text,
        "user_image":         user_image,
        "intent":             None,
        "clinical_evidence":  None,
        "retrieved_context":  None,
        "diagnosis_output":   None,
        "final_response":     "",
        "ablation_flags":     ablation_flags,       
    }
    
    # Route to the correct compiled app based on the flag
    if use_memory:
        result = app_with_memory.invoke(
            inputs,
            config={"configurable": {"thread_id": thread_id}},
        )
    else:
        # Stateless invocation for evaluation loops (ignores thread_id)
        result = app_without_memory.invoke(inputs)
        
    return result["final_response"]


# ================================================================
# Quick smoke test with Ablation Examples
# ================================================================
if __name__ == "__main__":
    
    print("=" * 60)
    print("Test 1: Standard Run (All Enabled)")
    print("=" * 60)
    r1 = run_turn("Can I take ibuprofen and aspirin together?")
    print(f"→ {r1}\n")

    print("=" * 60)
    print("Test 2: Ablation Run (GraphRAG Disabled)")
    print("=" * 60)
    r2 = run_turn(
        "Can I take ibuprofen and aspirin together?",
        ablation_flags={
            "use_vision": True,
            "use_diagnosis_agent": True,
            "use_medication_graphrag": False,  # Disabled!
            "use_general_vectorrag": True
        }
    )
    print(f"→ {r2}\n")

    print("=" * 60)
    print("Test 3: Ablation Run (Vision Agent Disabled)")
    print("=" * 60)
    r3 = run_turn(
        "I found this rash on my finger, what could it be?",
        user_image=str(PROJECT_ROOT / "processed/mmskin/MM-SkinQA-small/bk3_c8_43.png"),
        ablation_flags={
            "use_vision": False,               # Disabled! Image is ignored.
            "use_diagnosis_agent": True,
            "use_medication_graphrag": True,
            "use_general_vectorrag": True
        }
    )
    print(f"→ {r3}\n")
