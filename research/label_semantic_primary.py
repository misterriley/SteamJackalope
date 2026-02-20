import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys
import json
import torch
import re
from collections import Counter
from tqdm import tqdm

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    MODEL_NAME,
    W_DESC_FILE,
    ROOT_DIR,
    METADATA_FILE
)

def label_dimensions_from_descriptions():
    print("Loading model and whitening matrix...")
    device = "cuda" if torch.torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    if not os.path.exists(W_DESC_FILE):
        print(f"Error: {W_DESC_FILE} not found.")
        return
        
    W = np.load(W_DESC_FILE).astype(np.float32)
    num_dims = W.shape[1]
    
    # --- EXTRACT UNIQUE WORDS FROM DESCRIPTIONS ---
    print("Extracting unique words from short descriptions...")
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found.")
        return
        
    df = pd.read_parquet(METADATA_FILE, columns=['short_description'])
    all_text = " ".join(df['short_description'].fillna('').tolist()).lower()
    
    # Simple regex to get words (3+ characters)
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    
    # Count frequencies to filter out noise
    word_counts = Counter(words)
    # Keep top 10k words that appear at least 5 times
    vocab = [w for w, c in word_counts.most_common(10000) if c >= 5]
    
    print(f"Vocabulary size: {len(vocab)} unique words")
    
    # --- ENCODE AND PROJECT ---
    print("Encoding vocabulary...")
    embeddings_raw = model.encode(vocab, show_progress_bar=True, batch_size=128)
    
    print("Projecting into whitened space...")
    embeddings_whitened = np.dot(embeddings_raw, W)
    
    # Normalize for cosine similarity to dimensions
    norms = np.linalg.norm(embeddings_whitened, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    cosine_sims = embeddings_whitened / norms
    
    # --- FIND TOP WORD PER DIMENSION ---
    print("Finding primary word labels per dimension...")
    dimension_labels = {}
    
    for i in range(num_dims):
        loadings = cosine_sims[:, i]
        
        # Best positive match
        pos_idx = np.argmax(loadings)
        top_pos_word = vocab[pos_idx]
        top_pos_val = float(loadings[pos_idx])
        
        # Best negative match
        neg_idx = np.argmin(loadings)
        top_neg_word = vocab[neg_idx]
        top_neg_val = float(loadings[neg_idx])
        
        dimension_labels[str(i)] = {
            "primary_positive": top_pos_word,
            "primary_negative": top_neg_word,
            "pos_score": top_pos_val,
            "neg_score": top_neg_val,
            "context_pos": [vocab[idx] for idx in np.argsort(-loadings)[:5]],
            "context_neg": [vocab[idx] for idx in np.argsort(loadings)[:5]]
        }
        
    # --- SAVE RESULTS ---
    output_path = os.path.join(ROOT_DIR, "data", "production", "semantic_primary_labels.json")
    with open(output_path, 'w') as f:
        json.dump(dimension_labels, f, indent=4)
        
    print(f"Done! Saved primary labels to {output_path}")
    
    # Print a few examples
    print("\nSample Primary Dimension Labels:")
    for i in range(min(10, num_dims)):
        d = dimension_labels[str(i)]
        pos_word = d['primary_positive']
        neg_word = d['primary_negative']
        pos_score = d['pos_score']
        neg_score = d['neg_score']
        print(f"Dim {i:3}: {pos_word:15} vs {neg_word:15} (Scores: +{pos_score:.2f} / {neg_score:.2f})")

if __name__ == "__main__":
    label_dimensions_from_descriptions()
