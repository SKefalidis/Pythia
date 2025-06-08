import torch
import argparse
import sys
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def create_faiss_index(args):
    tqdm_kwargs = {"desc": "Creating documents", "file": sys.stdout}
    
    # ------------------------
    # ----- Prepare data -----
    # ------------------------
    print("Preparing data...")
    raw_data = open(args.input).read()
    texts = raw_data.split("\n")
    documents = [text for text in tqdm(texts, **tqdm_kwargs)] # need prefix, could also add this via SentenceTransformer

    # --------------------------
    # ----- Generate embeddings -----
    # --------------------------
    print("Generating embeddings...")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, device=device)
    
    embeddings = model.encode(
        documents, 
        show_progress_bar=True, 
        batch_size=512,
        device=device,
        prompt_name="passage"
    )

    embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32


    # --------------------------
    # ----- Create FAISS index -----
    # --------------------------
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ----------------------------------
    # ----- Save the index to disk -----
    # ----------------------------------
    print("Saving index to disk...")
    os.makedirs(args.output, exist_ok=True)
    faiss.write_index(index, f"{args.output}/faiss.index")
    
    with open(f"{args.output}/docs.txt", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

    print("Index generation complete!")
    print(f"Index and documents saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS Dense Index Generator")
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    create_faiss_index(args)
