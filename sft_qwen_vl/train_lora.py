import os
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image

# =========================
# Dataset & Collator
# =========================
class VLDataset(Dataset):
    """Dataset class for Qwen2-VL Fine-tuning."""
    def __init__(self, jsonl_path: str, image_dir: str, processor: AutoProcessor, nums_data: int = -1):
        full_dataset = load_dataset("json", data_files=jsonl_path)["train"]
        
        if nums_data > 0:
            actual_nums = min(nums_data, len(full_dataset))
            self.data = full_dataset.select(range(actual_nums))
        else:
            self.data = full_dataset
            
        self.image_dir = Path(image_dir)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        image_path = self.image_dir / sample["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path).convert("RGB")
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text, "image": image}


class VLDataCollator:
    """Callable data collator to handle dynamic batching and padding."""
    def __init__(self, processor: AutoProcessor, max_length: int, max_pixels: int):
        self.processor = processor
        self.max_length = max_length
        self.max_pixels = max_pixels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [feature["text"] for feature in features]
        images = [feature["image"] for feature in features]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.max_length,
            max_pixels=self.max_pixels 
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100

        return batch

# =========================
# Argument Parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU QLoRA Fine-tuning for Qwen2-VL")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Base model path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./qwen2_vl_lora", help="Output directory")
    parser.add_argument("--nums_data", type=int, default=1024, help="Number of data points used in fine-tuning")
    
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_pixels", type=int, default=518400, help="Max image pixels")
    
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=500, help="Save interval")

    return parser.parse_args()

# =========================
# Main Execution
# =========================
def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    if local_rank == 0:
        print(f"Loading processor: {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)

    train_dataset = VLDataset(
        jsonl_path=args.data_path, 
        image_dir=args.image_dir, 
        processor=processor,
        nums_data=args.nums_data
    )
    
    data_collator = VLDataCollator(
        processor=processor, 
        max_length=args.max_length, 
        max_pixels=args.max_pixels
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if local_rank == 0:
        print("Loading base model in 4-bit using SDPA...")
        
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", # Replaced flash_attention_2 with sdpa
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if local_rank == 0:
        print("Applying LoRA config...")
        
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True, 
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if local_rank == 0:
        print("Starting training...")
        
    trainer.train()

    if local_rank == 0:
        print(f"Saving fine-tuned model to {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print("Done!")

if __name__ == "__main__":
    main()