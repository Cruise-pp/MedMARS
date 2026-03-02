import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import evaluate

# =========================
# Argument Parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Qwen2-VL Model on Unseen Data")
    
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Base model name/path")
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to your saved LoRA weights")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing dataset images")
    
    # Matching the training script to split the dataset correctly
    parser.add_argument("--nums_data", type=int, default=1024, help="Number of samples that were used for training")
    
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate for the answer")
    
    return parser.parse_args()

# =========================
# Main Execution
# =========================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor from {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model)

    print("Loading base model in bfloat16...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print(f"Loading and merging LoRA weights from {args.lora_dir}...")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model = model.merge_and_unload()

    # Testing original model's result
    # model = base_model 
    model.eval()

    print("Loading evaluation dataset and splitting...")
    full_dataset = load_dataset("json", data_files=args.data_path)["train"]
    total_len = len(full_dataset)
    
    # ---------------------------------------------------------
    # Dataset Split Logic
    # ---------------------------------------------------------
    if 0 < args.nums_data < total_len:
        print(f"Train used {args.nums_data} samples.")
        print(f"Evaluation will use the remaining {total_len - args.nums_data} samples (Index {args.nums_data} to {total_len - 1}).")
        # Select everything from args.nums_data to the end of the dataset
        eval_data = full_dataset.select(range(args.nums_data, total_len))
    else:
        print("⚠️ Warning: --nums_data covers the whole dataset or is invalid. Using the entire dataset for evaluation.")
        eval_data = full_dataset

    image_dir = Path(args.image_dir)

    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("sacrebleu")

    predictions = []
    references = []

    print("Starting generation...")
    for sample in tqdm(eval_data, desc="Evaluating"):
        image_path = image_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        
        ground_truth = sample["answer"]
        question = sample["question"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True, 
        )

        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, 
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        prediction = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        print("*" * 50)
        print(f"Question: {question}")
        print(f"Model prediction: {prediction.strip()}")
        predictions.append(prediction.strip())
        references.append([ground_truth.strip()])

    print("Computing metrics...")
    flat_references = [ref[0] for ref in references] 
    
    # Compute ROUGE
    rouge_results = rouge_metric.compute(predictions=predictions, references=flat_references)
    
    # Compute BLEU
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)

    print("\n" + "="*40)
    print("🎯 Evaluation Results on Unseen Data")
    print("="*40)
    print(f"Total Samples evaluated: {len(predictions)}")
    print(f"ROUGE-1: {rouge_results['rouge1'] * 100:.2f}")
    print(f"ROUGE-2: {rouge_results['rouge2'] * 100:.2f}")
    print(f"ROUGE-L: {rouge_results['rougeL'] * 100:.2f}")
    print(f"BLEU:    {bleu_results['score']:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()