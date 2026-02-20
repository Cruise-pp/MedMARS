import os
import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen3-0.6B"
TRAIN_JSON = "processed/ddxplus_sft/train.jsonl"
VALID_JSON = "processed/ddxplus_sft/validate.jsonl"
OUT_DIR = "outputs/qwen06b_lora_ddxplus"

PROMPT_TMPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n"
)

def format_example(ex):
    text = PROMPT_TMPL.format(
        instruction=ex["instruction"],
        input=ex["input"],
    ) + ex["output"]
    return text


def main():
    if not os.path.exists(TRAIN_JSON):
        raise FileNotFoundError(TRAIN_JSON)

    has_valid = os.path.exists(VALID_JSON)

    # -----------------------------
    # 3) jsonl -> datasets
    # -----------------------------
    data_files = {"train": TRAIN_JSON}
    if has_valid:
        data_files["validation"] = VALID_JSON

    ds = load_dataset("json", data_files=data_files)

    # -----------------------------
    # 4) tokenizer / model
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # -----------------------------
    # 5) LoRA
    # -----------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        max_length=1024,
    )

    # -----------------------------
    # 7) Trainer
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if "validation" in ds else None,
        peft_config=lora_config,
        formatting_func=format_example,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"[OK] Training finished. Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()