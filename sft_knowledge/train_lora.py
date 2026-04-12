import os
import random
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_PROFILES = {
    "qwen_0.6b": {
        "model_name": "Qwen/Qwen-0.6B-Instruct",
        "out_dir": "outputs/qwen06b_lora_ddxplus",
        "batch_size": 32,
        "grad_accum": 1,
        "learning_rate": 1e-4,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "mistral_7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "out_dir": "outputs/mistral7b_lora_ddxplus",
        "batch_size": 4,
        "grad_accum": 8,
        "learning_rate": 2e-5,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
}

ACTIVE_PROFILE = "qwen_0.6b" # "mistral_7b"

# RUN_MODE:
#   - "full"   : full train split for 1 epoch
#   - "sample" : random subset for 1 epoch
RUN_MODE = "sample"

TRAIN_JSON = "processed/ddxplus_sft/train.jsonl"
VALID_JSON = "processed/ddxplus_sft/validate.jsonl"

MAX_LEN = 1024                          # 512 was too tight; DDXPlus cases can exceed it

# sample mode subset sizes
SUBSET_TRAIN = 100000
SUBSET_VALID = 10000

SEED = 42

PROMPT_TMPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n"
)


def format_example(ex):
    return PROMPT_TMPL.format(
        instruction=ex["instruction"],
        input=ex["input"],
    ) + ex["output"]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    # ---- load profile ----
    profile = MODEL_PROFILES[ACTIVE_PROFILE]
    model_name     = profile["model_name"]
    out_dir_base   = profile["out_dir"]
    batch_size     = profile["batch_size"]
    grad_accum     = profile["grad_accum"]
    learning_rate  = profile["learning_rate"]
    target_modules = profile["target_modules"]

    print(f"[Config] profile={ACTIVE_PROFILE} model={model_name} lr={learning_rate} "
          f"batch={batch_size}×{grad_accum} max_len={MAX_LEN}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    random.seed(SEED)

    if not os.path.exists(TRAIN_JSON):
        raise FileNotFoundError(TRAIN_JSON)
    has_valid = os.path.exists(VALID_JSON)

    data_files = {"train": TRAIN_JSON}
    if has_valid:
        data_files["validation"] = VALID_JSON
    ds = load_dataset("json", data_files=data_files)

    # subset selection
    if RUN_MODE == "sample":
        n_train = min(SUBSET_TRAIN, len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=SEED).select(range(n_train))
        if "validation" in ds:
            n_val = min(SUBSET_VALID, len(ds["validation"]))
            ds["validation"] = ds["validation"].shuffle(seed=SEED).select(range(n_val))
        out_dir = f"{out_dir_base}_sample"

    elif RUN_MODE == "full":
        out_dir = out_dir_base

    else:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE} (use full/sample)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # ---- eval strategy: run eval every 500 steps if validation set exists ----
    eval_kwargs = {}
    if has_valid:
        eval_kwargs["eval_strategy"] = "steps"
        eval_kwargs["eval_steps"] = 500

    args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=1,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        max_length=MAX_LEN,
        dataloader_num_workers=8,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        report_to=[],
        **eval_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if has_valid else None,
        peft_config=lora_config,
        formatting_func=format_example,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[DONE] Training finished. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
