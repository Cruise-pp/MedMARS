import os
import json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    csv_path = "../Datasets/MedQuAD.csv" 
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    cleaned_df = df.dropna(subset=['question', 'answer']).reset_index(drop=True)
    print(f"Data loaded. Valid QA pairs: {len(cleaned_df)}")

    texts_to_embed = "Question: " + cleaned_df['question'] + "\nAnswer: " + cleaned_df['answer']
    texts_to_embed = texts_to_embed.tolist()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Generating embeddings...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=64)

    os.makedirs("../processed/medquad", exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings) # type: ignore
    
    faiss.write_index(index, "../processed/medquad/medquad_index.faiss")

    with open("../processed/medquad/medquad_corpus.jsonl", "w", encoding="utf-8") as f:
        for i, row in cleaned_df.iterrows():
            record = {
                "id": i,
                "question": row["question"],
                "answer": row["answer"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("Build complete. FAISS index and JSONL corpus saved.")

if __name__ == "__main__":
    main()