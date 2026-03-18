"""
Preprocess data from the MMSkin clinical VQA dataset, save it as a JSONL file,
and create a sampled subset for fine-tuning processes.
"""
import json
import argparse
import shutil
import random
from pathlib import Path
import pandas as pd

random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent

def preprocess_mmskin_data(vqa_path: Path, caption_path: Path, output_path: Path) -> None:
    """
    Reads VQA and caption data, filters for the 'skin' modality, cleans image paths,
    and writes the formatted results to a JSONL file.
    """
    if not vqa_path.exists() or not caption_path.exists():
        raise FileNotFoundError(f"Could not find input files at {vqa_path} or {caption_path}")

    print(f"Loading data from {vqa_path.parent}...")
    df_vqa = pd.read_csv(vqa_path)
    df_caption = pd.read_csv(caption_path)

    # Merge, filter, and clean data using vectorized Pandas operations
    df_merged = pd.merge(df_vqa, df_caption[['image', 'modality']], on='image', how='inner')
    df_skin = df_merged[df_merged['modality'] == 'skin'].copy()
    
    # Clean image paths and select relevant columns
    df_skin['image'] = df_skin['image'].str.replace("dataset/", "", regex=False)
    records = df_skin[['image', 'question', 'answer']].to_dict(orient='records')

    # Ensure output directory exists and dump results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
            
    print(f"Preprocessing complete! Wrote {len(records)} records to {output_path}.")

def sample_data(data_path: Path, img_folder: Path, img_folder_small: Path, small_data_path: Path, sample_size: int = 1024) -> None:
    """
    Samples a specified number of records from the original JSONL and copies associated images.
    """
    with data_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Robust handling in case the dataset is smaller than the requested sample size
    actual_sample_size = min(sample_size, len(lines))
    if sample_size > len(lines):
        print(f"Warning: Requested sample size ({sample_size}) is larger than dataset. Taking all {len(lines)} records.")

    lines_sel = random.sample(lines, actual_sample_size)
    
    # Ensure destination image folder exists
    img_folder_small.mkdir(parents=True, exist_ok=True)

    with small_data_path.open('w', encoding='utf-8') as f:
        for line in lines_sel:
            # Write the raw JSON string directly instead of decoding/re-encoding
            f.write(line)
            
            # Extract image path to copy the file
            img_path = json.loads(line)["image"]
            src_img = img_folder / img_path
            dst_img = img_folder_small / img_path
            
            # Ensure subdirectories inside the image folder exist, then copy
            if src_img.exists():
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img) # copy2 preserves file metadata
            else:
                print(f"Warning: Image not found and skipped - {src_img}")

    print(f"Sampling complete! Wrote {actual_sample_size} records to {small_data_path}.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess and sample MMSkin clinical VQA data into JSONL format.")
    
    # Consolidated arguments
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "Datasets/MM-SkinQA",
                        help="Base directory containing VQA.csv, caption.csv, and original images.")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "processed/mmskin",
                        help="Base directory to save the JSONL files and sampled images.")
    parser.add_argument("--sample_size", type=int, default=1280,
                        help="Number of items to sample for the smaller dataset.")

    args = parser.parse_args()

    # Dynamically resolve paths based on base directories
    vqa_path = args.data_dir / "VQA.csv"
    caption_path = args.data_dir / "caption.csv"
    output_full = args.output_dir / "mmskin_sft.jsonl"
    
    small_data_path = args.output_dir / "mmskin_sft_small.jsonl"
    img_folder_small = args.output_dir / "MM-SkinQA-small"

    preprocess_mmskin_data(vqa_path, caption_path, output_full)
    sample_data(output_full, args.data_dir, img_folder_small, small_data_path, args.sample_size)

if __name__ == "__main__":
    main()
