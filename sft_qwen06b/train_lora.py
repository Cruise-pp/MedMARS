import os
import random
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# RUN_MODE:
#   - "full"   : full train split for 1 epoch
#   - "sample" : random subset (default 100k) for 1 epoch
#   - "quick"  : small subset + max_steps cap
RUN_MODE = "sample"

MODEL_NAME = "Qwen/Qwen3-0.6B"
TRAIN_JSON = "processed/ddxplus_sft/train.jsonl"
VALID_JSON = "processed/ddxplus_sft/validate.jsonl"
OUT_DIR = "outputs/qwen06b_lora_ddxplus"

# A100-friendly defaults
BATCH_SIZE = 32
GRAD_ACCUM = 1
MAX_LEN = 512

# quick mode
MAX_STEPS_QUICK = 2000
SUBSET_TRAIN_QUICK = 50000
SUBSET_VALID_QUICK = 5000

# sample mode
SUBSET_TRAIN_SAMPLE = 100000
SUBSET_VALID_SAMPLE = 10000

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
    if RUN_MODE == "quick":
        ds["train"] = ds["train"].select(range(min(SUBSET_TRAIN_QUICK, len(ds["train"]))))
        if "validation" in ds:
            ds["validation"] = ds["validation"].select(range(min(SUBSET_VALID_QUICK, len(ds["validation"]))))
        max_steps = MAX_STEPS_QUICK
        out_dir = f"{OUT_DIR}_quick"

    elif RUN_MODE == "sample":
        n_train = min(SUBSET_TRAIN_SAMPLE, len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=SEED).select(range(n_train))
        if "validation" in ds:
            n_val = min(SUBSET_VALID_SAMPLE, len(ds["validation"]))
            ds["validation"] = ds["validation"].shuffle(seed=SEED).select(range(n_val))
        max_steps = -1
        out_dir = f"{OUT_DIR}_sample"

    elif RUN_MODE == "full":
        max_steps = -1
        out_dir = OUT_DIR

    else:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE} (use full/sample/quick)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-4,
        max_steps=max_steps,
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
    )

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
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[DONE] Training finished. Saved to: {out_dir}")


if __name__ == "__main__":
    main()