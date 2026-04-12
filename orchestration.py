"""
Graph-VMA Orchestration Pipeline (Ablation-Ready)
=================================================
Multi-agent medical assistant built on LangGraph.

Architecture:
  Orchestrator (GPT-4o-mini intent classification + query rewrite)
    ├─ symptom_diagnosis  → Info Gathering (GPT) → KnowledgeAgent-Diagnosis (Mistral-7B)
    ├─ medication_inquiry → KnowledgeAgent-Medication (DrugBank GraphRAG + GPT)
    ├─ general_qa         → KnowledgeAgent-General    (MedQuAD VectorRAG + GPT)
    └─ has image?         → VisionAgent (Qwen2-VL) → KnowledgeAgent → Synthesizer
  All knowledge paths    → Synthesizer (GPT-4o-mini + rolling summary) → END
"""

import json
import time
import operator
import torch
import gc
import os
from pathlib import Path

# Load environment variables from .env file if it exists
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key not in os.environ:  # Don't override existing env vars
                    os.environ[key] = value

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    HumanMessage, SystemMessage, BaseMessage, AIMessage,
)
from drugbank_graph import drugbank_query as dq
from typing import TypedDict, Annotated, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from openai import OpenAI
from PIL import Image

# ================================================================
# Config
# ================================================================
# Maximum messages retained in state. Older messages are trimmed after each
# node update; their content is preserved in ``conversation_summary``.
# 20 messages ≈ 5-6 complete turns (Human + System + AI per turn).
MAX_RETAINED_MESSAGES = 20

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).resolve().parent

# DrugBank SQLite path (used by medication branch)
DRUGBANK_DB_PATH = os.getenv(
    "DRUGBANK_DB_PATH",
    str(PROJECT_ROOT / "processed/drugbank/drugbank_ddi.sqlite"),
)

# Fine-tuned model paths
DIAGNOSIS_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DIAGNOSIS_ADAPTER_DIR = os.getenv("DIAGNOSIS_ADAPTER_DIR", str(PROJECT_ROOT / "mistral7b_lora"))
VISION_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
VISION_ADAPTER_DIR = os.getenv("VISION_ADAPTER_DIR", str(PROJECT_ROOT / "qwen_vl_lora"))

# ================================================================
# 1. State
# ================================================================
def _add_and_trim_messages(
    left: List[BaseMessage], right: List[BaseMessage]
) -> List[BaseMessage]:
    """Custom reducer: append new messages, then keep only the most recent N.

    Older messages beyond ``MAX_RETAINED_MESSAGES`` are discarded. Their
    content is preserved in ``conversation_summary`` (rolling ~200 word
    summary updated by the Synthesizer every turn).
    """
    combined = left + right
    if len(combined) > MAX_RETAINED_MESSAGES:
        combined = combined[-MAX_RETAINED_MESSAGES:]
    return combined


class AgentState(TypedDict):
    """Shared state flowing through all LangGraph nodes.

    ``messages`` uses a custom reducer that appends and auto-trims to
    ``MAX_RETAINED_MESSAGES``, preventing unbounded growth. Older history
    is preserved in ``conversation_summary``.

    Fields without a reducer are overwritten each turn by ``run_turn()`` inputs,
    EXCEPT conversation_summary, gathering_rounds, and pending_followup_questions
    which are omitted from inputs so their checkpoint values persist across turns.
    """
    messages: Annotated[List[BaseMessage], _add_and_trim_messages]
    user_text: str
    user_image: Optional[str]
    intent: Optional[str]
    clinical_evidence: Optional[str]
    retrieved_context: Optional[str]
    diagnosis_output: Optional[str]
    final_response: str
    ablation_flags: dict                # Dictionary to control ablation
    conversation_summary: Optional[str] # Rolling summary of earlier turns
    gathering_rounds: Optional[int]     # Tracks info-gathering follow-up rounds
    retrieval_confidence: Optional[str] # "high" | "low" | "none" — set by knowledge agents
    safety_level: Optional[str]         # "safe" | "risky" | "emergency" | "self_harm"
    pending_followup_questions: Optional[List[str]]  # Follow-up Qs asked during gathering


# ================================================================
# 2. LLM helpers
# ================================================================
_openai_client = None

def _get_openai_client() -> OpenAI:
    """Return a singleton OpenAI client (avoids creating one per API call)."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "") or OPENAI_API_KEY
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def call_openai(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> str:
    """Call OpenAI's gpt-4o-mini model with exponential backoff retry."""
    import openai as _openai

    client = _get_openai_client()
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
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
        except (
            _openai.RateLimitError,
            _openai.APITimeoutError,
            _openai.APIConnectionError,
            _openai.InternalServerError,
        ) as e:
            last_exc = e
            if attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[call_openai] {type(e).__name__}, retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)

    raise last_exc


# ================================================================
# 2b. Chat history helper (multi-turn support)
# ================================================================
def _format_chat_history(
    messages: List[BaseMessage],
    max_turns: int = 3,
    max_chars: int = 2000,
    summary: str = "",
) -> str:
    """Format conversation context for prompt injection.

    Structure: [Summary of earlier turns] + [Recent N raw turn-pairs].
    Keeps only HumanMessage/AIMessage (drops SystemMessage bookkeeping),
    excludes the last HumanMessage (current turn), and truncates to
    stay within token budget.  Returns "" when there is no prior history.
    """
    relevant = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    # Drop the last HumanMessage — that is the *current* turn
    if relevant and isinstance(relevant[-1], HumanMessage):
        relevant = relevant[:-1]

    if not relevant and not summary:
        return ""

    # Keep only the most recent N turn-pairs as raw messages
    recent = relevant[-(max_turns * 2):]

    sections = []

    # Prepend summary if available (covers earlier turns beyond recent window)
    if summary:
        sections.append(f"[Summary of earlier conversation]\n{summary}")

    # Append recent raw messages
    if recent:
        lines = []
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        sections.append("[Recent messages]\n" + "\n".join(lines))

    history = "\n\n".join(sections)

    # Hard truncation from the front (keep recent context)
    if len(history) > max_chars:
        history = history[-max_chars:]
        idx = history.find("\n")
        if idx != -1:
            history = history[idx + 1:]
        history = "...\n" + history

    return history


# ================================================================
# 2c. Rolling conversation summary
# ================================================================
_SUMMARY_SYSTEM = """\
You are a medical conversation summarizer. Produce a concise summary of the
conversation so far, focusing on clinically relevant information.

Include: chief complaints, confirmed diagnoses (use exact labels when available),
medications mentioned, drug interactions found, key advice given,
and any unresolved questions.
Keep the summary under 200 words. Reply with ONLY the summary."""


def _extract_clinical_note(diagnosis_output: str) -> str:
    """Extract a one-line clinical label from structured diagnosis_output JSON.

    Returns a short note (e.g. "Diagnosis: Migraine. Differential: ...") for
    the summarizer to preserve exact clinical terms. Returns "" when no
    structured labels are available (e.g. general QA).
    """
    try:
        data = json.loads(diagnosis_output)
    except (json.JSONDecodeError, TypeError):
        return ""

    # Diagnosis agent
    if "primary_diagnosis" in data and data.get("status") == "final":
        note = f"Diagnosis: {data['primary_diagnosis']}"
        ddx = data.get("differential_diagnosis", [])
        if ddx:
            labels = [d["label"] for d in ddx[:3] if isinstance(d, dict) and "label" in d]
            if labels:
                note += f". Differential: {', '.join(labels)}"
        return note

    # Medication agent
    if "drugs_identified" in data:
        drugs = [d["name"] for d in data["drugs_identified"] if isinstance(d, dict)]
        note = f"Drugs discussed: {', '.join(drugs)}" if drugs else ""
        if data.get("has_interactions"):
            note += ". Drug interactions found"
        if data.get("unresolved"):
            note += f". Unresolved: {', '.join(data['unresolved'])}"
        return note

    return ""


def _update_summary(
    prev_summary: str,
    user_text: str,
    assistant_response: str,
    clinical_note: str = "",
) -> str:
    """Generate an updated rolling summary (~200 words) incorporating the latest exchange.

    When ``clinical_note`` is provided (extracted from structured diagnosis_output),
    it is included so the summary preserves exact diagnosis labels and drug names
    rather than only the patient-friendly paraphrase.
    """
    parts = []
    if prev_summary:
        parts.append(f"[Previous summary]\n{prev_summary}")
    parts.append(f"[Latest exchange]\nUser: {user_text}\nAssistant: {assistant_response}")
    if clinical_note:
        parts.append(f"[Key clinical findings]\n{clinical_note}")
    prompt = "\n\n".join(parts) + "\n\n[Updated summary]"
    return call_openai(_SUMMARY_SYSTEM, prompt, max_tokens=300)


# ================================================================
# 2d. Safety guardrail
# ================================================================
import re

_SELF_HARM_PATTERNS = re.compile(
    r"(?i)(想死|不想活|自杀|自残|割腕|跳楼|结束生命|活不下去|"
    r"kill\s*myself|suicide|end\s*(my|it\s*all)|overdose\s*on|"
    r"want\s*to\s*die|self[- ]?harm|cut\s*myself|jump\s*off)",
)
_EMERGENCY_PATTERNS = re.compile(
    r"(?i)(大量出血|无法呼吸|晕倒|失去意识|心脏骤停|"
    r"chest\s*pain.*can'?t\s*breathe|severe\s*bleeding|"
    r"unconscious|not\s*breathing|seizure\s*right\s*now|"
    r"heart\s*attack|choking|anaphylax)",
)

_SAFETY_CLASSIFY_SYSTEM = """\
You are a medical safety classifier. Classify the safety level of this patient message.

Labels (reply with EXACTLY one):
  EMERGENCY  — life-threatening situation, needs immediate emergency services (call 911/120)
  SELF_HARM  — expresses suicidal ideation, self-harm intent, or desire to end their life
  RISKY      — asks about potentially dangerous self-treatment (stopping prescribed meds,
               taking someone else's prescription, dangerous drug combinations for harm)
  SAFE       — normal medical question with no safety concern

Reply with the label ONLY (one word, no explanation)."""

_EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency.\n\n"
    "Please take immediate action:\n"
    "- Call 120 (China) / 911 (US) / 999 (UK) immediately\n"
    "- If someone is with you, ask them for help\n"
    "- Stay as calm as possible while waiting for emergency services\n\n"
    "I am an AI assistant and cannot provide emergency medical care. "
    "Please contact professional medical services right away."
)

_SELF_HARM_RESPONSE = (
    "I hear you, and what you're feeling matters.\n\n"
    "Please reach out to a professional who can help:\n"
    "- National 24h Crisis Hotline (China): 400-161-9995\n"
    "- Beijing Crisis Center: 010-82951332\n"
    "- National Suicide Prevention Lifeline (US): 988\n"
    "- Crisis Text Line (US): Text HOME to 741741\n\n"
    "You don't have to face this alone. "
    "Please talk to someone you trust — a friend, family member, or counselor."
)

_RISKY_WARNING = (
    "\n\n---\n*Important safety notice: The action you described could be dangerous. "
    "Please do NOT adjust medications, dosages, or treatments without consulting "
    "your doctor or pharmacist first.*"
)


def _check_safety(user_text: str) -> str:
    """Two-layer safety check: fast keyword match, then LLM classification.

    Returns one of: "emergency", "self_harm", "risky", "safe".
    """
    # Layer 1: fast keyword regex (zero API cost)
    if _SELF_HARM_PATTERNS.search(user_text):
        print("[Safety] SELF_HARM detected via keyword match")
        return "self_harm"
    if _EMERGENCY_PATTERNS.search(user_text):
        print("[Safety] EMERGENCY detected via keyword match")
        return "emergency"

    # Layer 2: LLM classification (catches subtle/indirect expressions)
    raw = call_openai(_SAFETY_CLASSIFY_SYSTEM, user_text, max_tokens=20)
    label = raw.strip().upper().replace(" ", "_")

    if label in ("EMERGENCY", "SELF_HARM", "RISKY"):
        print(f"[Safety] {label} detected via LLM classification")
        return label.lower()

    return "safe"


# ================================================================
# 3. Orchestrator  — intent classification via gpt
# ================================================================
_INTENT_SYSTEM = """\
You are a medical triage router. Classify the user query into EXACTLY ONE label:

  symptom_diagnosis   – user describes symptoms / complaints / asks for diagnosis
  medication_inquiry  – user asks about drugs, drug interactions, side effects, dosage
  general_qa          – general medical knowledge question
  chitchat            – greetings, thanks, farewells, small talk, or non-medical chatter
                        (e.g. "hello", "thank you", "bye", "how are you", "ok thanks")

If conversation history is provided, use it to resolve references
(e.g., "that drug", "those symptoms") and correctly classify the follow-up.
Reply with the label ONLY (one line, no quotes, no explanation)."""

_VALID_INTENTS = {"symptom_diagnosis", "medication_inquiry", "general_qa", "chitchat"}

_GATHERING_ARBITRATE_SYSTEM = """\
You are a dialogue flow controller for a medical assistant that is currently
collecting symptom information from the patient.

The system previously asked the patient follow-up questions about their symptoms.
Determine if the patient's new message is:

  CONTINUE - answering the follow-up questions, providing more symptom details,
             or continuing the diagnostic conversation (even if they mention
             medications they have tried as part of describing their situation)
  SWITCH   - explicitly asking about a completely different topic that is
             unrelated to the ongoing symptom assessment

Reply with EXACTLY one word: CONTINUE or SWITCH"""


def _arbitrate_gathering_intent(
    user_text: str,
    pending_questions: list[str],
    conversation_summary: str,
) -> str:
    """Decide whether a user message continues the gathering dialogue or switches topic.

    Returns "CONTINUE" or "SWITCH". Defaults to "CONTINUE" if the LLM
    response is unparseable (fail-safe: don't break the gathering flow).
    """
    parts = []
    if conversation_summary:
        parts.append(f"[Conversation so far]\n{conversation_summary}")
    if pending_questions:
        parts.append(
            "[Follow-up questions the system just asked]\n"
            + "\n".join(f"- {q}" for q in pending_questions)
        )
    parts.append(f"[Patient's new message]\n{user_text}")
    prompt = "\n\n".join(parts)

    raw = call_openai(_GATHERING_ARBITRATE_SYSTEM, prompt, max_tokens=10)
    decision = raw.strip().upper()

    if decision in ("CONTINUE", "SWITCH"):
        return decision
    print(f"[Arbitrator] Unparseable response '{raw}', defaulting to CONTINUE")
    return "CONTINUE"

_REWRITE_SYSTEM = """\
You are a query rewriter. Given a conversation history and a follow-up message,
rewrite the follow-up into a standalone, self-contained query.

Rules:
- Resolve all pronouns and references (e.g., "it", "that drug", "those symptoms")
  using the conversation history
- Keep the rewritten query concise and natural
- If the message is already self-contained, return it unchanged
- Reply with ONLY the rewritten query, nothing else"""

def _rewrite_query(user_text: str, history: str) -> str:
    """Rewrite a vague follow-up into a self-contained query.

    Example: history mentions diabetes → "How is it treated?" becomes
    "How is type 2 diabetes treated?"
    """
    prompt = (
        f"[Conversation history]\n{history}\n\n"
        f"[Follow-up message]\n{user_text}\n\n"
        f"[Rewritten query]"
    )
    rewritten = call_openai(_REWRITE_SYSTEM, prompt, max_tokens=256)
    print(f"[Orchestrator] query rewrite: '{user_text}' -> '{rewritten}'")
    return rewritten

def orchestrator(state: AgentState) -> dict:
    """Safety check → gathering arbitration → intent classification → query rewrite.

    First runs a two-layer safety check (keyword + LLM). If the query is
    classified as EMERGENCY or SELF_HARM, sets ``safety_level`` and
    ``intent="safety_blocked"`` to short-circuit the pipeline. RISKY queries
    proceed normally but carry a flag for the Synthesizer to append a warning.

    When ``gathering_rounds > 0`` (active info-gathering for diagnosis),
    an arbitration gate determines whether the user is continuing the
    diagnostic dialogue or switching to a new topic. This prevents
    false-positive intent switches (e.g., "I took ibuprofen" being
    misclassified as medication_inquiry when user is answering symptom
    follow-up questions).

    For safe queries, classifies intent into one of three knowledge branches
    and rewrites vague follow-ups into self-contained queries.
    """
    user_text = state.get("user_text", "")
    if not user_text.strip():
        return {
            "intent": "general_qa",
            "safety_level": "safe",
            "gathering_rounds": 0,
            "pending_followup_questions": [],
        }

    # ── Safety check (runs before everything else) ──
    safety = _check_safety(user_text)
    if safety in ("emergency", "self_harm"):
        return {
            "intent": "safety_blocked",
            "safety_level": safety,
            "messages": [SystemMessage(content=f"[Safety] blocked: {safety}")],
        }

    # ── Gathering arbitration (context stickiness) ──
    gathering_rounds = state.get("gathering_rounds") or 0
    if gathering_rounds > 0:
        pending_qs = state.get("pending_followup_questions") or []
        summary_for_arb = state.get("conversation_summary", "") or ""
        decision = _arbitrate_gathering_intent(user_text, pending_qs, summary_for_arb)
        print(f"[Orchestrator] Gathering active (round {gathering_rounds}), arbitration={decision}")

        if decision == "CONTINUE":
            # Force intent to symptom_diagnosis, skip normal classification
            # Still rewrite the query for context resolution
            history = _format_chat_history(state.get("messages", []), summary=summary_for_arb)
            rewritten = _rewrite_query(user_text, history) if history else user_text
            return {
                "intent": "symptom_diagnosis",
                "user_text": rewritten,
                "safety_level": safety,
                "messages": [SystemMessage(content="[Orchestrator] intent=symptom_diagnosis (gathering continuation)")],
            }
        else:
            # SWITCH: user explicitly changed topic — reset gathering state
            print(f"[Orchestrator] User switched topic during gathering, resetting gathering_rounds")
            # Fall through to normal classification below

    # ── Normal intent classification ──
    summary = state.get("conversation_summary", "") or ""
    history = _format_chat_history(state.get("messages", []), summary=summary)
    if history:
        prompt = (
            f"[Recent conversation]\n{history}\n\n"
            f"[Current message]\n{user_text}"
        )
        # Rewrite vague follow-ups into self-contained queries
        rewritten = _rewrite_query(user_text, history)
    else:
        prompt = user_text
        rewritten = user_text

    raw = call_openai(_INTENT_SYSTEM, prompt)
    intent = raw.strip().strip('"').strip("'").lower().replace(" ", "_")

    if intent not in _VALID_INTENTS:
        intent = "general_qa"

    print(f"[Orchestrator] intent = {intent}")
    result = {
        "intent": intent,
        "user_text": rewritten,  # downstream agents get the self-contained query
        "safety_level": safety,  # "safe" or "risky"
        "messages": [SystemMessage(content=f"[Orchestrator] intent={intent}")],
    }

    # Reset gathering state if user switched topic mid-gathering
    if gathering_rounds > 0:
        result["gathering_rounds"] = 0
        result["pending_followup_questions"] = []

    return result


# ================================================================
# Memory Manager
# ================================================================
_active_gpu_model = None

def _manage_vram(target_model_name: str):
    """Swap models between CPU RAM and GPU VRAM.

    Only one 7B model fits in VRAM at a time (Mistral-7B or Qwen2-VL-7B).
    This function offloads the inactive model to CPU, flushes the CUDA cache,
    then loads the target model to GPU. No-op if the target is already on GPU.
    """
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
    """Lazy-load Qwen2-VL-7B base model + LoRA adapter, merged, to CPU."""
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
    """Analyze a medical image using fine-tuned Qwen2-VL-7B.

    Processes the image at ``state["user_image"]`` with the user's text prompt,
    producing free-text ``clinical_evidence`` (e.g. "eczema with erythema and scaling").
    This evidence is stored in state for downstream knowledge agents to use.
    """
    _ensure_vision_model_loaded()
    _manage_vram("vision")

    image_path = state.get("user_image", "")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Vision Agent] Failed to load image '{image_path}': {e}")
        return {"clinical_evidence": ""}
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
        print(f"[Vision Agent] CUDA OOM: {e}")
        return {"clinical_evidence": ""}

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
    """Lazy-load Mistral-7B-Instruct base model + LoRA adapter to CPU."""
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

def _extract_symptoms(user_text: str, clinical_evidence: str = "", conversation_summary: str = "") -> dict:
    """Extract structured symptom data from free text via GPT.

    Combines user text, conversation summary, and vision-derived clinical
    evidence into a single prompt.  Returns a dict with keys:
    ``age``, ``sex``, ``symptoms`` (list), ``antecedents`` (list).
    Falls back to empty values on parse failure.
    """
    parts = []
    if conversation_summary:
        parts.append(f"[Conversation history summary]\n{conversation_summary}")
    parts.append(f"[Current message]\n{user_text}")
    if clinical_evidence:
        parts.append(f"[Clinical evidence from imaging]\n{clinical_evidence}")
    prompt = "\n\n".join(parts)

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
    """Build the Mistral-7B SFT inference prompt from extracted symptom data.

    Formats age, sex, symptoms, and antecedents into the ``### Instruction / ### Input / ### Output``
    template that the fine-tuned LoRA adapter expects (trained on DDXPlus).
    """
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
    """Run Mistral-7B inference on the formatted SFT prompt.

    Assumes ``_diag_model`` and ``_diag_tokenizer`` are already loaded.
    Returns the raw generated text (decoded, special tokens stripped).
    """
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
    """Extract a JSON object from Mistral's raw output.

    Finds the first ``{...}`` block and parses it.  Returns a dict with
    ``primary_diagnosis`` and ``differential_diagnosis`` keys, falling back
    to ``"unknown"`` / ``[]`` on parse failure.
    """
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

_COMPLETENESS_SYSTEM = """\
You are a medical triage assistant. Assess whether the patient has provided enough
information for a preliminary diagnosis.

Minimum required information:
- At least 2-3 specific symptoms (not just "I feel bad")
- Some characterization of symptoms (location, nature, duration, severity)
- Basic patient context (age/sex if relevant, or enough symptoms to compensate)

Reply with EXACTLY one of:
  COMPLETE - enough information to attempt a diagnosis
  INCOMPLETE - need more details, list 1-2 specific questions to ask

Format your reply as:
COMPLETE
or
INCOMPLETE
Q1: [specific follow-up question]
Q2: [specific follow-up question]"""

MAX_GATHERING_ROUNDS = 2

def _assess_completeness(user_text: str, conversation_summary: str, clinical_evidence: str = "") -> dict:
    """Assess whether enough symptom info exists for a preliminary diagnosis.

    Uses GPT to evaluate the combined context (summary + current message +
    clinical evidence). Returns ``{"complete": True, "questions": []}`` if
    sufficient, or ``{"complete": False, "questions": [...]}`` with 1-2
    specific follow-up questions when more info is needed.
    """
    parts = []
    if conversation_summary:
        parts.append(f"[Conversation so far]\n{conversation_summary}")
    parts.append(f"[Current message]\n{user_text}")
    if clinical_evidence:
        parts.append(f"[Clinical evidence from imaging]\n{clinical_evidence}")
    prompt = "\n\n".join(parts)

    raw = call_openai(_COMPLETENESS_SYSTEM, prompt, max_tokens=200)
    lines = raw.strip().split("\n")

    if lines[0].strip().upper().startswith("COMPLETE"):
        return {"complete": True, "questions": []}

    questions = []
    for line in lines[1:]:
        line = line.strip()
        if line.startswith(("Q1:", "Q2:")):
            questions.append(line[3:].strip())
    return {"complete": False, "questions": questions}


def knowledge_diagnosis(state: AgentState) -> dict:
    """Symptom-to-diagnosis pipeline with info-gathering dialogue.

    Three-phase flow:
    1. **Completeness gate** — GPT assesses if enough symptoms are present.
       If incomplete and under ``MAX_GATHERING_ROUNDS``, returns follow-up
       questions (status="gathering") without invoking Mistral.
    2. **Ablation check** — if ``use_diagnosis_agent=False``, uses GPT
       fallback instead of the fine-tuned model.
    3. **Mistral inference** — extracts structured symptoms, formats the
       SFT prompt, and runs the LoRA-adapted Mistral-7B for differential
       diagnosis. Resets ``gathering_rounds`` to 0 after final output.
    """
    flags = state.get("ablation_flags", {})
    user_text = state.get("user_text", "")
    clinical_evidence = state.get("clinical_evidence", "")
    conversation_summary = state.get("conversation_summary", "") or ""
    gathering_rounds = state.get("gathering_rounds") or 0

    # ── Step 1: Info completeness gate (always runs, uses GPT not Mistral) ──
    if gathering_rounds < MAX_GATHERING_ROUNDS:
        assessment = _assess_completeness(user_text, conversation_summary, clinical_evidence)

        if not assessment["complete"]:
            gathering_rounds += 1
            print(f"[Knowledge-Diagnosis] Info incomplete (round {gathering_rounds}/{MAX_GATHERING_ROUNDS}), gathering more.")
            questions_text = "\n".join(f"- {q}" for q in assessment["questions"]) if assessment["questions"] else ""
            return {
                "diagnosis_output": json.dumps({
                    "status": "gathering",
                    "round": gathering_rounds,
                    "max_rounds": MAX_GATHERING_ROUNDS,
                    "follow_up_questions": assessment["questions"],
                    "message": f"Need more information to make a diagnosis.\n{questions_text}",
                }, ensure_ascii=False),
                "gathering_rounds": gathering_rounds,
                "retrieval_confidence": "high",  # gathering is intentional, not a confidence issue
                "pending_followup_questions": assessment["questions"],
            }

    # ── Step 2: Info sufficient → check ablation ──
    if not flags.get("use_diagnosis_agent", True):
        print("[Knowledge-Diagnosis] Ablation: Using GPT fallback for diagnosis.")
        # GPT-based fallback when Mistral is unavailable
        fallback_prompt = f"Based on: {conversation_summary}\nCurrent: {user_text}"
        if clinical_evidence:
            fallback_prompt += f"\nClinical evidence: {clinical_evidence}"
        answer = call_openai(
            "You are a medical assistant. Provide a brief preliminary assessment based on the symptoms described. Always recommend consulting a healthcare professional.",
            fallback_prompt, max_tokens=512,
        )
        return {
            "diagnosis_output": json.dumps({
                "status": "final",
                "primary_diagnosis": "GPT-based assessment (no fine-tuned model)",
                "assessment": answer,
                "differential_diagnosis": [],
            }, ensure_ascii=False),
            "gathering_rounds": 0,
            "retrieval_confidence": "high",
            "pending_followup_questions": [],
        }

    # ── Step 3: Run Mistral (info complete + model available) ──
    print(f"[Knowledge-Diagnosis] Info sufficient. Running diagnosis model.")
    print(f"[Knowledge-Diagnosis] query='{user_text[:80]}'")

    extracted = _extract_symptoms(user_text, clinical_evidence, conversation_summary)
    prompt = _format_sft_prompt(extracted)

    _ensure_diagnosis_model_loaded()
    _manage_vram("diagnosis")

    raw_output = _run_diagnosis_inference(prompt)
    diagnosis = _parse_diagnosis_json(raw_output)
    diagnosis["status"] = "final"
    diagnosis_output = json.dumps(diagnosis, ensure_ascii=False)

    # Reset gathering_rounds after final diagnosis
    return {"diagnosis_output": diagnosis_output, "gathering_rounds": 0, "retrieval_confidence": "high", "pending_followup_questions": []}


# ================================================================
# 5b. Knowledge Agent — Medication  (GraphRAG + gpt)
# ================================================================

_DRUG_EXTRACT_SYSTEM = """\
Extract all drug / medication names from the user query.
Return a JSON array of strings, e.g. ["ibuprofen", "aspirin"].
If no drugs are mentioned, return [].
Reply with ONLY the JSON array, nothing else."""

def _extract_drug_names(user_text: str) -> list[str]:
    """Extract drug/medication names from free text via GPT.

    Returns a list of cleaned drug name strings, or an empty list
    if no drugs are mentioned or parsing fails.
    """
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
    """Drug information and interaction lookup via DrugBank GraphRAG.

    Extracts drug names from user text (via GPT), resolves each against
    the DrugBank SQLite database (exact → prefix → fuzzy match), retrieves
    indications/descriptions, and checks all pairwise drug-drug interactions.
    Returns structured JSON with identified drugs, interactions, and a
    human-readable ``retrieved_context`` for the Synthesizer.
    """
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

    # ── Retrieval confidence: based on resolve success rate ──
    total_drugs = len(drug_names)
    resolved_count = len(resolved)
    if total_drugs == 0:
        confidence = "none"
    elif resolved_count == total_drugs:
        confidence = "high"
    elif resolved_count > 0:
        confidence = "low"
    else:
        confidence = "none"
    print(f"[Knowledge-Medication] confidence={confidence} (resolved {resolved_count}/{total_drugs})")

    return {
        "retrieved_context": retrieved_context,
        "diagnosis_output":  diagnosis_output,
        "retrieval_confidence": confidence,
    }

# ================================================================
# 5c. Knowledge Agent — General QA  (MedQuAD RAG + gpt)
# ================================================================
_GENERAL_QA_SYSTEM = """\
You are a medical knowledge assistant. Answer the patient's question accurately
based ONLY on the provided reference materials.

Rules:
- Base your answer strictly on the provided context; do not fabricate information
- If the context does not fully answer the question, say so honestly
- Use clear, professional language
- You MUST cite references using [Ref N] tags (e.g., [Ref 1], [Ref 2]) for every
  factual claim you make. If a statement cannot be tied to a reference, do not include it."""

# RRF confidence parameters
_RRF_SCORE_FLOOR = 0.010              # Absolute minimum: below this, no result is trustworthy
_RRF_CONCENTRATION_THRESHOLD = 1.3    # top1/mean ratio: above this, top-1 is an isolated outlier

def knowledge_general(state: AgentState) -> dict:
    """General medical Q&A via MedQuAD hybrid retrieval (FAISS + BM25).

    Searches the MedQuAD knowledge base using the user's query, enriched
    with ``clinical_evidence`` from the Vision Agent when available.
    Feeds the top-5 retrieved Q&A pairs as context to GPT for answer
    generation.  Supports ablation (``use_general_vectorrag=False``)
    which skips retrieval and answers via raw LLM.
    """
    user_text = state.get("user_text", "")
    clinical_evidence = state.get("clinical_evidence", "")
    flags = state.get("ablation_flags", {})

    if not flags.get("use_general_vectorrag", True):
        print("[Knowledge-General] Ablation: Skipping VectorRAG, answering via raw LLM.")
        answer = call_openai(_GENERAL_QA_SYSTEM, f"Patient question: {user_text}", max_tokens=1024)
        return {
            "retrieved_context": "VectorRAG Ablated",
            "diagnosis_output":  json.dumps({"answer": answer}, ensure_ascii=False)
        }

    from medquad_rag import query_index as mq

    # Enrich search query with vision output when available
    # e.g. "What is this?" + "eczema with redness" → search "eczema redness skin condition"
    search_query = user_text
    if clinical_evidence:
        search_query = f"{user_text} {clinical_evidence}"
        print(f"[Knowledge-General] enriched query with clinical evidence")

    print(f"[Knowledge-General] query='{search_query[:80]}'")
    results = mq.search(search_query, top_k=5)

    # ── Retrieval confidence gate (absolute floor + concentration ratio) ──
    top_score = results[0]["score"] if results else 0.0
    mean_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    concentration = top_score / mean_score if mean_score > 0 else float("inf")

    if not results or top_score < _RRF_SCORE_FLOOR:
        confidence = "none"
    elif concentration > _RRF_CONCENTRATION_THRESHOLD:
        confidence = "low"     # top-1 is an isolated match, rest are noise
    elif top_score < 0.020 and concentration < 1.1:
        confidence = "low"     # uniformly low scores — all results are noise
    else:
        confidence = "high"    # multiple results agree → retrieval is reliable
    print(f"[Knowledge-General] top={top_score:.4f}, mean={mean_score:.4f}, "
          f"concentration={concentration:.2f} → confidence={confidence}")

    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Reference {i+1}] (score={r['score']:.4f})\n"
            f"Q: {r['question']}\n"
            f"A: {r['answer']}"
        )
    retrieved_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

    if confidence == "none":
        answer = ("I could not find relevant medical references for your question. "
                  "Please consult a healthcare professional for accurate guidance.")
    elif confidence == "low":
        # Low-confidence: still show results but add explicit caveat
        user_prompt = ""
        if clinical_evidence:
            user_prompt += f"Clinical evidence from image analysis:\n{clinical_evidence}\n\n"
        user_prompt += (
            f"Reference materials (NOTE: these references may not be highly relevant):\n"
            f"{retrieved_context}\n\n"
            f"Patient question: {user_text}\n\n"
            f"Answer based on the references. If the references do not adequately "
            f"address the question, clearly state that your information is limited:"
        )
        answer = call_openai(_GENERAL_QA_SYSTEM, user_prompt, max_tokens=1024)
    else:
        user_prompt = ""
        if clinical_evidence:
            user_prompt += f"Clinical evidence from image analysis:\n{clinical_evidence}\n\n"
        user_prompt += (
            f"Reference materials:\n{retrieved_context}\n\n"
            f"Patient question: {user_text}\n\n"
            f"Provide a comprehensive answer based on the references above:"
        )
        answer = call_openai(_GENERAL_QA_SYSTEM, user_prompt, max_tokens=1024)

    diagnosis_output = json.dumps({"answer": answer}, ensure_ascii=False)

    return {
        "retrieved_context": retrieved_context,
        "diagnosis_output":  diagnosis_output,
        "retrieval_confidence": confidence,
    }


# ================================================================
# 5d. Faithfulness check (post-generation NLI)
# ================================================================
_FAITHFULNESS_SYSTEM = """\
You are a medical fact-checker. Given a CONTEXT (retrieved knowledge) and a RESPONSE
(generated answer), evaluate whether each claim in the response is supported by the context.

For each sentence in the response, label it as:
  SUPPORTED     — the claim is directly backed by information in the context
  NOT_SUPPORTED — the claim contains information not found in the context
  NEUTRAL       — the sentence is a greeting, disclaimer, or recommendation (not a factual claim)

Reply with a JSON object:
{
  "verdict": "FAITHFUL" or "UNFAITHFUL",
  "flagged_sentences": ["sentence that is not supported", ...],
  "supported_ratio": <float between 0 and 1>
}

Rules:
- A response is FAITHFUL if ALL factual claims are SUPPORTED or NEUTRAL
- Generic medical advice like "consult a doctor" is NEUTRAL (not flagged)
- Be strict: if a specific drug name, dosage, or diagnosis is mentioned but not in the context, flag it
- Reply with ONLY the JSON object"""


def _check_faithfulness(context: str, response: str) -> dict:
    """Check whether each factual claim in response is grounded in context.

    Returns a dict with ``verdict`` ("FAITHFUL"/"UNFAITHFUL"),
    ``flagged_sentences`` (list of unsupported claims), and
    ``supported_ratio`` (float 0-1).

    On parse failure, returns UNFAITHFUL as a fail-safe — for medical content,
    "unable to verify" should be treated as "not verified".
    """
    if not context or not response:
        return {"verdict": "FAITHFUL", "flagged_sentences": [], "supported_ratio": 1.0}

    prompt = (
        f"[CONTEXT]\n{context[:3000]}\n\n"
        f"[RESPONSE]\n{response}\n\n"
        f"[Evaluation]"
    )
    raw = call_openai(_FAITHFULNESS_SYSTEM, prompt, max_tokens=512)
    cleaned = raw.strip().strip("`").strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    print(f"[Faithfulness] Failed to parse LLM response, defaulting to UNFAITHFUL")
    return {"verdict": "UNFAITHFUL", "flagged_sentences": [], "supported_ratio": 0.0}


# ================================================================
# 6. Synthesizer  (gpt)
# ================================================================
_SYNTH_SYSTEM = """\
You are a compassionate medical assistant conducting a diagnostic conversation.
Convert the clinical reasoning below into a clear, empathetic response.

Rules:
- Use simple language the patient can understand
- Always recommend seeing a healthcare professional for serious concerns
- Be empathetic but factual; do NOT fabricate information
- If conversation history is provided, maintain continuity and avoid repeating info
- When citing retrieved knowledge, use [Ref N] tags to indicate the source

Confidence-aware rules:
- If the retrieval confidence is "low" or "none", explicitly tell the patient that
  your information on this topic is limited and they should verify with a professional.
  Do NOT present low-confidence information as definitive fact.

Diagnostic dialogue rules:
- If the clinical reasoning indicates "status: gathering" (still collecting info),
  acknowledge the patient's input, briefly explain what you understand so far,
  and ask the follow-up questions provided in the clinical reasoning.
  Do NOT attempt a diagnosis yet — just gather more information.
- If the clinical reasoning contains a final diagnosis ("status: final"),
  present the results clearly and recommend seeing a healthcare professional.
- Keep follow-up questions clinically relevant and easy for patients to answer.

[Example doctor responses — match this style and tone]

Example 1 (Diagnosis):
Patient: I had a sonogram that revealed 2 liver cysts. Now I have pain in my right side. I'm 61 and afraid of surgery.
Doctor: Hi, I am sorry to hear you are having pain. Cysts in the liver are a fairly \
frequent finding on USG scans. Most often these cysts tend to be "simple cysts" and \
do not cause trouble. Some other causes include hereditary diseases and infections, \
but in those conditions patients usually have symptoms from the disease. In your case, \
it appears to be a simple cyst. Although mostly asymptomatic, simple cysts can produce \
symptoms when 1) they enlarge to a big size, 2) when there is bleeding into the cyst, \
or 3) when they develop an infection. Since both cysts are small, I would recommend a \
followup scan at a later date to see if the cyst is enlarging. Hope this helps and \
hope you start to feel better soon.

Example 2 (Medication):
Patient: My 3-year-old son has been prescribed Budecort 200 (steroid) two puffs twice daily for asthma. I'm uncomfortable with steroid use at this age.
Doctor: Hi, I admire your positive outlook regarding the query. For asthma management, \
physicians need to advise various medications depending on severity. Levolin acts by \
dilating air passages and relieves acute breathlessness. Budecort, which is a steroid, \
acts by reducing inflammation. It has some side effects like fungal infections of the \
mouth (Candidiasis). Your son has been prescribed a total of 800 micrograms, which \
equals only 0.8 mg. Moreover, very little steroid is absorbed via inhalation. \
Therefore, side effects are almost negligible except candidiasis — that is why your \
child is advised to gurgle his mouth with plain water after taking puffs. Hope this \
answers your query. Wishing your son good health.

Example 3 (General QA):
Patient: My wife was diagnosed with TB. What precautions should we take? She stopped breastfeeding our daughter.
Doctor: Hi, I understand your concern. First, you need to get her fully investigated \
by a chest physician to know the extent of her infection — whether it is active TB \
(passing infection to others via coughing) requiring isolation, or non-infecting and \
under control. If non-infecting, she can be treated at home and can feed her baby. \
Medicines must be taken regularly, and follow-up is a must. TB can be fully controlled \
if she follows the treatment plan exactly. Apart from medication: 1) high-protein diet \
— milk 2 glasses a day, boiled eggs, fresh vegetables and salads; 2) at least 10 \
glasses of clean water; 3) light exercises such as yoga, walking, and respiratory \
exercises to help appetite and digestion; 4) iron and vitamins as per doctor's advice. \
Thanks."""

_DISCLAIMER_LOW_CONFIDENCE = (
    "\n\n---\n*Note: The information above is based on limited reference matches "
    "and may not be fully accurate. Please consult a qualified healthcare "
    "professional for reliable medical advice.*"
)
_DISCLAIMER_UNFAITHFUL = (
    "\n\n---\n*Note: Parts of this response could not be fully verified against "
    "our medical knowledge base. Please consult a healthcare professional.*"
)


# ── Chitchat node (bypasses RAG entirely) ──

_CHITCHAT_SYSTEM = """\
You are a friendly medical assistant. The user sent a casual message (greeting,
thanks, farewell, or small talk). Reply naturally in 1-2 sentences. Be warm but
brief. If they seem to want medical help, invite them to ask a health question.
You MUST reply in English. Do not use any other language."""


def chitchat_response(state: AgentState) -> dict:
    """Handle greetings, thanks, and small talk without RAG retrieval."""
    user_text = state.get("user_text", "")
    prev_summary = state.get("conversation_summary", "") or ""
    history = _format_chat_history(state.get("messages", []), summary=prev_summary)

    prompt = ""
    if history:
        prompt += f"[Conversation history]\n{history}\n\n"
    prompt += f"User: {user_text}"

    response = call_openai(_CHITCHAT_SYSTEM, prompt, max_tokens=128)

    new_summary = _update_summary(prev_summary, user_text, response, "")

    return {
        "final_response": response,
        "messages": [AIMessage(content=response)],
        "conversation_summary": new_summary,
    }


def synthesizer(state: AgentState) -> dict:
    """Generate a patient-friendly response with faithfulness verification.

    Combines clinical evidence, retrieved knowledge, and diagnosis output
    into a context block, injects conversation history for continuity,
    and calls GPT to produce an empathetic, plain-language response.
    After generation, runs a faithfulness check (NLI) against the retrieved
    context and appends disclaimers when confidence is low or claims are
    unsupported.  Also generates an updated rolling summary (~200 words).
    """
    diagnosis_output  = state.get("diagnosis_output",  "")
    clinical_evidence = state.get("clinical_evidence",  "")
    retrieved_context = state.get("retrieved_context",  "")
    user_text         = state.get("user_text", "")
    confidence        = state.get("retrieval_confidence", "high") or "high"

    # Inject conversation history (summary + recent messages) for coherent responses
    prev_summary = state.get("conversation_summary", "") or ""
    history = _format_chat_history(state.get("messages", []), summary=prev_summary)

    parts = []
    if clinical_evidence:
        parts.append(f"[Visual Findings]\n{clinical_evidence}")
    if retrieved_context:
        parts.append(f"[Retrieved Knowledge]\n{retrieved_context}")
    if diagnosis_output:
        parts.append(f"[Clinical Reasoning]\n{diagnosis_output}")

    context = "\n\n".join(parts) if parts else "(no context)"

    user_prompt = ""
    if history:
        user_prompt += f"[Conversation history]\n{history}\n\n"
    user_prompt += (
        f"[Retrieval confidence: {confidence}]\n\n"
        f"Patient query: {user_text}\n\n"
        f"Clinical context:\n{context}\n\n"
        f"Generate a patient-friendly response:"
    )

    final_response = call_openai(_SYNTH_SYSTEM, user_prompt)

    # ── Post-generation faithfulness check ──
    if retrieved_context and confidence != "none":
        faithfulness = _check_faithfulness(retrieved_context, final_response)
        verdict = faithfulness.get("verdict", "FAITHFUL")
        flagged = faithfulness.get("flagged_sentences", [])
        ratio = faithfulness.get("supported_ratio", 1.0)
        print(f"[Synthesizer] faithfulness={verdict}, supported_ratio={ratio:.2f}, flagged={len(flagged)}")

        if verdict == "UNFAITHFUL" and flagged:
            final_response += _DISCLAIMER_UNFAITHFUL
    else:
        print(f"[Synthesizer] faithfulness check skipped (no retrieved context)")

    # ── Low-confidence disclaimer ──
    if confidence == "low":
        final_response += _DISCLAIMER_LOW_CONFIDENCE

    # ── Risky query safety warning ──
    if state.get("safety_level") == "risky":
        final_response += _RISKY_WARNING

    # Generate rolling summary (compresses full conversation into ~200 words)
    clinical_note = _extract_clinical_note(diagnosis_output)
    new_summary = _update_summary(prev_summary, user_text, final_response, clinical_note)
    print(f"[Synthesizer] response length = {len(final_response)}")
    print(f"[Synthesizer] summary updated ({len(new_summary)} chars)")

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "conversation_summary": new_summary,
    }


# ================================================================
# 7. Routing functions
# ================================================================
_INTENT_TO_NODE = {
    "symptom_diagnosis":  "knowledge_diagnosis",
    "medication_inquiry": "knowledge_medication",
    "general_qa":         "knowledge_general",
    "chitchat":           "chitchat_response",
}

def route_after_orchestrator(state: AgentState) -> str:
    """Conditional edge: safety exit, Vision Agent, or Knowledge Agent.

    If the safety check flagged the query (emergency / self_harm), routes to
    ``safety_exit`` which returns a hardcoded safe response and skips the
    entire knowledge pipeline.  Otherwise routes to vision or knowledge.
    """
    # Safety-blocked queries bypass the entire pipeline
    if state.get("intent") == "safety_blocked":
        return "safety_exit"

    flags = state.get("ablation_flags", {})

    # Check if Vision is disabled via ablation flags
    if state.get("user_image") and flags.get("use_vision", True):
        return "vision_agent"

    intent = state.get("intent", "general_qa")
    return _INTENT_TO_NODE.get(intent, "knowledge_general")


def route_after_vision(state: AgentState) -> str:
    """Conditional edge: route to a Knowledge Agent after vision analysis.

    After the Vision Agent produces ``clinical_evidence``, this routes
    to the knowledge node matching the intent so that downstream agents
    can incorporate the visual findings into their retrieval/reasoning.
    """
    intent = state.get("intent", "general_qa")
    node = _INTENT_TO_NODE.get(intent, "knowledge_general")
    print(f"[Router] Vision done → {node} (intent={intent})")
    return node


# ================================================================
# 8. Build LangGraph
# ================================================================

def safety_exit(state: AgentState) -> dict:
    """Return a hardcoded safety response for emergency/self-harm queries.

    Bypasses all knowledge agents and synthesizer — uses fixed templates
    to avoid any risk of LLM-generated harmful content.
    """
    level = state.get("safety_level", "emergency")
    if level == "self_harm":
        response = _SELF_HARM_RESPONSE
    else:
        response = _EMERGENCY_RESPONSE
    print(f"[Safety Exit] Returning {level} template response")

    user_text = state.get("user_input", "")
    prev_summary = state.get("conversation_summary", "")
    clinical_note = f"SAFETY EVENT: {level} detected"
    new_summary = _update_summary(prev_summary, user_text, response, clinical_note)

    return {
        "final_response": response,
        "messages": [AIMessage(content=response)],
        "conversation_summary": new_summary,
        "gathering_rounds": 0,
        "pending_followup_questions": [],
        # Clear stale knowledge state so next turn doesn't inherit old context
        "retrieved_context": "",
        "diagnosis_output": "",
        "clinical_evidence": "",
        "retrieval_confidence": None,
        "safety_level": None,
    }


workflow = StateGraph(AgentState)

workflow.add_node("orchestrator",         orchestrator)
workflow.add_node("safety_exit",          safety_exit)
workflow.add_node("chitchat_response",    chitchat_response)
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
        "safety_exit":          "safety_exit",
        "chitchat_response":    "chitchat_response",
        "vision_agent":         "vision_agent",
        "knowledge_diagnosis":  "knowledge_diagnosis",
        "knowledge_medication": "knowledge_medication",
        "knowledge_general":    "knowledge_general",
    },
)

# After vision analysis, route to knowledge agent based on intent
workflow.add_conditional_edges(
    "vision_agent",
    route_after_vision,
    {
        "knowledge_diagnosis":  "knowledge_diagnosis",
        "knowledge_medication": "knowledge_medication",
        "knowledge_general":    "knowledge_general",
    },
)

workflow.add_edge("knowledge_diagnosis",  "synthesizer")
workflow.add_edge("knowledge_medication", "synthesizer")
workflow.add_edge("knowledge_general",    "synthesizer")
workflow.add_edge("synthesizer", END)
workflow.add_edge("safety_exit", END)
workflow.add_edge("chitchat_response", END)

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
    use_memory: bool = True,
) -> str:
    """Execute one conversation turn through the full LangGraph pipeline.

    Args:
        user_text: The user's natural-language query.
        user_image: Optional path to a medical image for the Vision Agent.
        thread_id: Conversation thread ID for MemorySaver checkpointing.
        ablation_flags: Dict toggling individual agents on/off for ablation studies.
        use_memory: If True, uses ``app_with_memory`` (stateful, multi-turn).
            If False, uses ``app_without_memory`` (stateless, for evaluation).

    Returns:
        The final patient-friendly response string.

    Note:
        ``conversation_summary``, ``gathering_rounds``, and
        ``pending_followup_questions`` are intentionally omitted from inputs
        so their checkpoint values persist across turns.
    """
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
        "intent":             "",
        "clinical_evidence":  "",
        "retrieved_context":  "",
        "diagnosis_output":   "",
        "final_response":     "",
        "ablation_flags":     ablation_flags,
        "retrieval_confidence": "",
        "safety_level":       "",
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
    r1 = run_turn("Can I take ibuprofen and aspirin together?", thread_id="multi_turn_demo")
    print(f"→ {r1}\n")

    print("=" * 60)
    print("Test 2: Multi-Turn Follow-Up (same thread)")
    print("=" * 60)
    r2 = run_turn("What about the side effects of those drugs?", thread_id="multi_turn_demo")
    print(f"→ {r2}\n")

    print("=" * 60)
    print("Test 3: Ablation Run (GraphRAG Disabled)")
    print("=" * 60)
    r3 = run_turn(
        "Can I take ibuprofen and aspirin together?",
        thread_id="ablation_test",
        ablation_flags={
            "use_vision": True,
            "use_diagnosis_agent": True,
            "use_medication_graphrag": False,  # Disabled!
            "use_general_vectorrag": True
        }
    )
    print(f"→ {r3}\n")

    print("=" * 60)
    print("Test 4: Ablation Run (Vision Agent Disabled)")
    print("=" * 60)
    r4 = run_turn(
        "I found this rash on my finger, what could it be?",
        user_image=str(PROJECT_ROOT / "processed/mmskin/MM-SkinQA-small/bk3_c8_43.png"),
        thread_id="ablation_vision",
        ablation_flags={
            "use_vision": False,               # Disabled! Image is ignored.
            "use_diagnosis_agent": True,
            "use_medication_graphrag": True,
            "use_general_vectorrag": True
        }
    )
    print(f"→ {r4}\n")
