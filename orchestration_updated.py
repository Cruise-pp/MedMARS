"""
Graph-VMA Orchestration Pipeline
=================================
Multi-agent medical assistant built on LangGraph.

Architecture:
  Orchestrator (DeepSeek V3 intent classification)
    ├─ symptom_diagnosis  → KnowledgeAgent-Diagnosis (fine-tuned LLM)
    ├─ medication_inquiry → KnowledgeAgent-Medication (GraphRAG + LLM)
    ├─ general_qa         → KnowledgeAgent-General    (RAG + LLM)
    └─ has image?         → VisionAgent → KnowledgeAgent-*
  All knowledge paths    → Synthesizer (DeepSeek V3) → END
"""

import requests
import json
import operator
import torch


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    HumanMessage, SystemMessage, BaseMessage, AIMessage,
)
from medquad_rag import query_index as mq
from drugbank_graph import drugbank_query as dq
from typing import TypedDict, Annotated, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================================================================
# Config
# ================================================================
SILICONFLOW_API_KEY = "sk-xxx"                          # TODO: 换成你的 key
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3"

# DrugBank SQLite path (used by medication branch)
DRUGBANK_DB_PATH = "processed/drugbank/drugbank_ddi.sqlite"


# ================================================================
# 1. State
# ================================================================
class AgentState(TypedDict):
    # ---- conversation history (auto-accumulates via operator.add) ----
    messages: Annotated[List[BaseMessage], operator.add]

    # ---- current-turn inputs ----
    user_text: str
    user_image: Optional[str]           # file path or base64; None = no image

    # ---- orchestrator output ----
    intent: Optional[str]               # symptom_diagnosis | medication_inquiry | general_qa

    # ---- vision agent output ----
    clinical_evidence: Optional[str]    # structured signals extracted from image

    # ---- retrieval output ----
    retrieved_context: Optional[str]    # GraphRAG / vector-RAG snippets

    # ---- knowledge agent output ----
    diagnosis_output: Optional[str]     # structured reasoning (JSON string)

    # ---- synthesizer output ----
    final_response: str


# ================================================================
# 2. LLM helpers
# ================================================================
def call_deepseek(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call DeepSeek-V3 via SiliconFlow (OpenAI-compatible endpoint)."""

    resp = requests.post(
        f"{SILICONFLOW_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ================================================================
# 3. Orchestrator  — intent classification via DeepSeek
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

    raw = call_deepseek(_INTENT_SYSTEM, user_text)
    intent = raw.strip().strip('"').strip("'").lower().replace(" ", "_")

    if intent not in _VALID_INTENTS:
        intent = "general_qa"

    print(f"[Orchestrator] intent = {intent}")
    return {
        "intent": intent,
        "messages": [SystemMessage(content=f"[Orchestrator] intent={intent}")],
    }


# ================================================================
# 4. Vision Agent  (STUB — Qwen2-VL-7B-Instruct)
# ================================================================
def vision_agent(state: AgentState) -> dict:
    """
    TODO:
      1. Load image from state["user_image"]
      2. Send to Qwen2-VL-7B-Instruct
      3. Extract structured clinical signals
         e.g. {"region": "left forearm", "finding": "erythematous plaque", "size": "3cm"}
      4. Return as clinical_evidence string for downstream Knowledge Agent
    """
    image_path = state.get("user_image", "")
    print(f"[Vision Agent] Processing image: {image_path}  (STUB)")

    clinical_evidence = (
        "[STUB] Visual findings extracted from image — "
        "to be replaced with Qwen2-VL output."
    )
    return {"clinical_evidence": clinical_evidence}


# ================================================================
# 5a. Knowledge Agent — Diagnosis  (STUB — fine-tuned Mistral-7B)
# ================================================================

# Config
DIAGNOSIS_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DIAGNOSIS_ADAPTER_DIR = "outputs/mistral7b_lora_ddxplus_sample"

# Lazy-loaded model
_diag_model = None
_diag_tokenizer = None

def _ensure_diagnosis_model_loaded():
    global _diag_model, _diag_tokenizer
    if _diag_model is not None:
        return

    print("[Knowledge-Diagnosis] Loading base model + LoRA adapter...")
    _diag_tokenizer = AutoTokenizer.from_pretrained(
        DIAGNOSIS_BASE_MODEL, trust_remote_code=True
    )
    if _diag_tokenizer.pad_token is None:
        _diag_tokenizer.pad_token = _diag_tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        DIAGNOSIS_BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _diag_model = PeftModel.from_pretrained(base, DIAGNOSIS_ADAPTER_DIR)
    _diag_model.eval()
    print("[Knowledge-Diagnosis] Model loaded.")


# ---- Step 1: Symptom extraction via DeepSeek ----
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
    """Extract structured symptoms from free text via DeepSeek."""
    prompt = user_text
    if clinical_evidence:
        prompt += f"\n\nAdditional clinical evidence from imaging:\n{clinical_evidence}"

    raw = call_deepseek(_SYMPTOM_EXTRACT_SYSTEM, prompt)
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


# ---- Step 2: Format into SFT prompt ----
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
    """Convert extracted symptoms into the same format as ddxplus_sft.py training data."""
    age = extracted.get("age")
    sex = extracted.get("sex")
    symptoms = extracted.get("symptoms", [])
    antecedents = extracted.get("antecedents", [])

    # Header
    age_str = str(age) if age is not None else "unknown"
    sex_str = str(sex) if sex is not None else "unknown"
    header = f"Patient: age={age_str}, sex={sex_str}"

    # Symptoms section
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

# ---- Step 3: Model inference + JSON parsing ----
def _run_diagnosis_inference(prompt: str, max_new_tokens: int = 512) -> str:
    """Run inference with the fine-tuned LoRA model."""
    assert _diag_model is not None and _diag_tokenizer is not None

    _ensure_diagnosis_model_loaded()

    inputs = _diag_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )
    inputs = {k: v.to(_diag_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = _diag_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_diag_tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return _diag_tokenizer.decode(gen_ids, skip_special_tokens=True)

def _parse_diagnosis_json(raw: str) -> dict:
    """Extract and parse JSON from model output."""
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

# ---- Main node function ----
def knowledge_diagnosis(state: AgentState) -> dict:
    """
    Diagnosis pipeline:
      1. Extract structured symptoms from user_text (DeepSeek)
      2. Format into SFT prompt (same as training data format)
      3. Run fine-tuned Mistral-7B LoRA inference
      4. Parse JSON output → primary_diagnosis + differential_diagnosis
    """
    user_text = state.get("user_text", "")
    clinical_evidence = state.get("clinical_evidence", "")

    assert clinical_evidence is not None
    print(f"[Knowledge-Diagnosis] query='{user_text[:80]}'")

    # step 1: extract symptoms
    extracted = _extract_symptoms(user_text, clinical_evidence)
    print(f"[Knowledge-Diagnosis] extracted: age={extracted.get('age')} "
          f"sex={extracted.get('sex')} symptoms={len(extracted.get('symptoms', []))}")

    # step 2: format prompt
    prompt = _format_sft_prompt(extracted)

    # step 3: run inference
    raw_output = _run_diagnosis_inference(prompt)
    print(f"[Knowledge-Diagnosis] raw output: {raw_output[:120]}")

    # step 4: parse
    diagnosis = _parse_diagnosis_json(raw_output)
    diagnosis_output = json.dumps(diagnosis, ensure_ascii=False)

    print(f"[Knowledge-Diagnosis] primary={diagnosis.get('primary_diagnosis', '?')}")
    return {"diagnosis_output": diagnosis_output}


# ================================================================
# 5b. Knowledge Agent — Medication  (GraphRAG + DeepSeek)
# ================================================================

_DRUG_EXTRACT_SYSTEM = """\
Extract all drug / medication names from the user query.
Return a JSON array of strings, e.g. ["ibuprofen", "aspirin"].
If no drugs are mentioned, return [].
Reply with ONLY the JSON array, nothing else."""

def _extract_drug_names(user_text: str) -> list[str]:
    """Ask DeepSeek to pull drug names from free text."""
    raw = call_deepseek(_DRUG_EXTRACT_SYSTEM, user_text)
    # defensive parse: strip markdown fences if model wraps them
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
    """
    Medication inquiry pipeline:
      1. Extract drug names from user_text  (DeepSeek)
      2. Resolve each name → DrugBank ID    (drugbank_query)
      3. Retrieve DDI between drug pairs     (drugbank_query)
      4. Fetch individual drug info          (drugbank_query)
      5. Assemble retrieved_context + diagnosis_output
    """

    user_text = state.get("user_text", "")
    print(f"[Knowledge-Medication] query='{user_text[:80]}'")

    # ---- step 1: extract drug names ----
    drug_names = _extract_drug_names(user_text)
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

    # ---- step 2: resolve each drug name ----
    resolved: list[dict] = []          # successfully resolved drugs
    failed:   list[str]  = []          # names that couldn't be resolved

    for name in drug_names:
        r = dq.resolve(name, db_path=DRUGBANK_DB_PATH)
        if r["status"] == "not_found":
            failed.append(name)
        else:
            # take the top candidate
            top = r["candidates"][0]
            drug_info = dq.get_drug(top["drug_id"], db_path=DRUGBANK_DB_PATH)
            resolved.append({
                "query":       name,
                "drug_id":     top["drug_id"],
                "name":        top["name"],
                "indication":  (drug_info.get("drug") or {}).get("indication", ""),
                "description": (drug_info.get("drug") or {}).get("description", ""),
            })

    # ---- step 3: check DDI between all resolved pairs ----
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

    # ---- step 4: build retrieved_context (for synthesizer) ----
    ctx_parts: list[str] = []

    for d in resolved:
        lines = f"Drug: {d['name']} ({d['drug_id']})"
        if d["indication"]:
            lines += f"\n  Indication: {d['indication'][:300]}"
        ctx_parts.append(lines)

    for inter in interactions:
        header = f"Interaction: {inter['drug_a']} ↔ {inter['drug_b']}"
        evidence_text = "\n  ".join(inter["evidence"][:3])   # cap at 3 to save tokens
        ctx_parts.append(f"{header}\n  {evidence_text}")

    if failed:
        ctx_parts.append(f"Unresolved drugs: {', '.join(failed)}")

    retrieved_context = "\n\n".join(ctx_parts)

    # ---- step 5: build diagnosis_output ----
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

    print(f"[Knowledge-Medication] resolved={len(resolved)} failed={len(failed)} interactions={len(interactions)}")
    return {
        "retrieved_context": retrieved_context,
        "diagnosis_output":  diagnosis_output,
    }


# ================================================================
# 5c. Knowledge Agent — General QA  (MedQuAD RAG + DeepSeek)
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
    """
    General medical QA pipeline:
      1. Retrieve top-k relevant QA pairs from MedQuAD (FAISS + BM25)
      2. Assemble retrieved context
      3. Call DeepSeek to generate answer grounded in context
    """
    from medquad_rag import query_index as mq

    user_text = state.get("user_text", "")
    print(f"[Knowledge-General] query='{user_text[:80]}'")

    # ---- step 1: retrieve from MedQuAD ----
    results = mq.search(user_text, top_k=5)
    print(f"[Knowledge-General] retrieved {len(results)} results"
          + (f", top score={results[0]['score']:.4f}" if results else ""))

    # ---- step 2: assemble retrieved context ----
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Reference {i+1}]\n"
            f"Q: {r['question']}\n"
            f"A: {r['answer']}"
        )
    retrieved_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

    # ---- step 3: generate answer via DeepSeek ----
    if retrieved_context:
        user_prompt = (
            f"Reference materials:\n{retrieved_context}\n\n"
            f"Patient question: {user_text}\n\n"
            f"Provide a comprehensive answer based on the references above:"
        )
        answer = call_deepseek(_GENERAL_QA_SYSTEM, user_prompt, max_tokens=1024)
    else:
        answer = ("I could not find relevant medical references for your question. "
                  "Please consult a healthcare professional for accurate guidance.")

    diagnosis_output = json.dumps({"answer": answer}, ensure_ascii=False)

    print(f"[Knowledge-General] answer length = {len(answer)}")
    return {
        "retrieved_context": retrieved_context,
        "diagnosis_output":  diagnosis_output,
    }


# ================================================================
# 6. Synthesizer  (STUB — DeepSeek V3)
# ================================================================
_SYNTH_SYSTEM = """\
You are a compassionate medical assistant. Convert the clinical reasoning below
into a clear, empathetic, and safety-aware response for the patient.

Rules:
- Use simple language the patient can understand
- Always recommend seeing a healthcare professional for serious concerns
- Be empathetic but factual; do NOT fabricate information"""


def synthesizer(state: AgentState) -> dict:
    diagnosis_output  = state.get("diagnosis_output",  "")
    clinical_evidence = state.get("clinical_evidence",  "")
    retrieved_context = state.get("retrieved_context",  "")
    user_text         = state.get("user_text", "")

    # assemble context block for the synthesizer prompt
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

    # TODO: uncomment to use real DeepSeek API
    # final_response = call_deepseek(_SYNTH_SYSTEM, user_prompt)
    final_response = f"[STUB Synthesizer] Based on: {diagnosis_output[:120]}…" # type: ignore

    print(f"[Synthesizer] response length = {len(final_response)}")
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
    }


# ================================================================
# 7. Routing functions
# ================================================================
def route_after_orchestrator(state: AgentState) -> str:
    """
    Orchestrator → ?
      - If image present  → vision_agent (then vision routes to knowledge)
      - Else              → knowledge_* based on intent
    """
    if state.get("user_image"):
        return "vision_agent"

    intent = state.get("intent", "general_qa")
    _INTENT_TO_NODE = {
        "symptom_diagnosis":  "knowledge_diagnosis",
        "medication_inquiry": "knowledge_medication",
        "general_qa":         "knowledge_general",
    }
    return _INTENT_TO_NODE[intent] if intent in _INTENT_TO_NODE else "knowledge_general"


def route_after_vision(state: AgentState) -> str:
    """
    Vision Agent → Knowledge Agent (decided by intent).
    Image signals are already stored in clinical_evidence.
    """
    intent = state.get("intent", "symptom_diagnosis")
    _INTENT_TO_NODE = {
        "symptom_diagnosis":  "knowledge_diagnosis",
        "medication_inquiry": "knowledge_medication",
        "general_qa":         "knowledge_general",
    }
    return _INTENT_TO_NODE[intent] if intent in _INTENT_TO_NODE else "knowledge_diagnosis"


# ================================================================
# 8. Build LangGraph
# ================================================================
memory = MemorySaver()
workflow = StateGraph(AgentState)

# -- nodes --
workflow.add_node("orchestrator",         orchestrator)
workflow.add_node("vision_agent",         vision_agent)
workflow.add_node("knowledge_diagnosis",  knowledge_diagnosis)
workflow.add_node("knowledge_medication", knowledge_medication)
workflow.add_node("knowledge_general",    knowledge_general)
workflow.add_node("synthesizer",          synthesizer)

# -- entry --
workflow.set_entry_point("orchestrator")

# -- orchestrator routing --
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

# -- vision → knowledge routing --
workflow.add_conditional_edges(
    "vision_agent",
    route_after_vision,
    {
        "knowledge_diagnosis":  "knowledge_diagnosis",
        "knowledge_medication": "knowledge_medication",
        "knowledge_general":    "knowledge_general",
    },
)

# -- all knowledge → synthesizer --
workflow.add_edge("knowledge_diagnosis",  "synthesizer")
workflow.add_edge("knowledge_medication", "synthesizer")
workflow.add_edge("knowledge_general",    "synthesizer")

# -- synthesizer → END --
workflow.add_edge("synthesizer", END)

# -- compile --
app = workflow.compile(checkpointer=memory)


# ================================================================
# 9. Convenience runner (for testing)
# ================================================================
def run_turn(
    user_text: str,
    user_image: str | None = None,
    thread_id: str = "default",
) -> str:
    """Run one user turn and return the final response."""
    inputs: AgentState = {                          # type: ignore[typeddict-item]
        "messages":           [HumanMessage(content=user_text)],
        "user_text":          user_text,
        "user_image":         user_image,
        "intent":             None,
        "clinical_evidence":  None,
        "retrieved_context":  None,
        "diagnosis_output":   None,
        "final_response":     "",
    }
    result = app.invoke(
        inputs,
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["final_response"]


# ================================================================
# Quick smoke test
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Symptom query")
    print("=" * 60)
    r1 = run_turn("I have a persistent cough and mild fever for 3 days.")
    print(f"→ {r1}\n")

    print("=" * 60)
    print("Test 2: Medication query")
    print("=" * 60)
    r2 = run_turn("Can I take ibuprofen and aspirin together?")
    print(f"→ {r2}\n")

    print("=" * 60)
    print("Test 3: General QA")
    print("=" * 60)
    r3 = run_turn("What is type 2 diabetes?")
    print(f"→ {r3}\n")

    print("=" * 60)
    print("Test 4: Image + symptom")
    print("=" * 60)
    r4 = run_turn(
        "I found this rash on my arm, what could it be?",
        user_image="test_rash.jpg",
    )
    print(f"→ {r4}\n")