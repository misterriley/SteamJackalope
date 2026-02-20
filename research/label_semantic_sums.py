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

def label_dimensions_with_word_sums():
    print("Loading model and whitening matrix...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    if not os.path.exists(W_DESC_FILE):
        print(f"Error: {W_DESC_FILE} not found.")
        return
        
    W = np.load(W_DESC_FILE).astype(np.float32)
    num_dims = W.shape[1]
    
    # --- GATHER 10,000 WORD VOCABULARY ---
    print("Extracting 10,000 words from short descriptions...")
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found.")
        return
        
    df = pd.read_parquet(METADATA_FILE, columns=['short_description'])
    all_text = " ".join(df['short_description'].fillna('').tolist()).lower()
    
    # Extract only single words (no spaces, 3+ chars)
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    word_counts = Counter(words)
    
    # Take the top 10,000 unique single words
    vocab = [w for w, c in word_counts.most_common(10000)]
    print(f"Vocabulary size: {len(vocab)} unique single words")
    
    # --- ENCODE AND PROJECT ---
    print("Encoding vocabulary...")
    embeddings_raw = model.encode(vocab, show_progress_bar=True, batch_size=128)
    
    print("Projecting into whitened space...")
    V = np.dot(embeddings_raw, W).astype(np.float32) # (N, D)
    
    # --- FIND ALIGNED SUMS (OPTIMIZED) ---
    print("Finding aligned sums for each dimension (Top-K Candidate method)...")
    
    output_data = {}
    K = 500 # Number of candidate words to check per side of each dimension
    
    for d in tqdm(range(num_dims), desc="Labeling Dimensions"):
        # 1. Get projections for this dimension
        projections = V[:, d]
        
        # 2. Get top K positive and top K negative candidates
        pos_indices = np.argsort(-projections)[:K]
        neg_indices = np.argsort(projections)[:K]
        
        def find_best_sum(indices):
            # Subset vectors: (K, D)
            sub_v = V[indices]
            # All pairs sum: (K, K, D)
            # We can vectorize this but even better: just do the math
            # Maximize (A.d + B.d) / ||A + B||
            
            # Sums of projections (Numerators): (K, K)
            num = projections[indices][:, np.newaxis] + projections[indices][np.newaxis, :]
            
            # Sums of vectors (for Denominator calculation): (K, K, D)
            # Actually, we only need the Norms squared: ||A+B||^2 = ||A||^2 + ||B||^2 + 2(A.B)
            # Precompute norms squared
            norms_sq = np.sum(sub_v**2, axis=1) # (K,)
            # Dots: (K, K)
            dots = np.dot(sub_v, sub_v.T)
            den_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] + 2 * dots
            den = np.sqrt(np.maximum(den_sq, 1e-9))
            
            # Scores: (K, K)
            scores = num / den
            
            # Find best pair
            max_idx = np.argmax(np.abs(scores))
            r, c = np.unravel_index(max_idx, scores.shape)
            return vocab[indices[r]], vocab[indices[c]], float(scores[r, c])

        p1, p2, p_score = find_best_sum(pos_indices)
        n1, n2, n_score = find_best_sum(neg_indices)
        
        p_label = f"{p1} + {p2}" if p1 != p2 else p1
        n_label = f"{n1} + {n2}" if n1 != n2 else n1
        
        output_data[str(d)] = {
            "positive_label": p_label,
            "negative_label": n_label,
            "dynamic_label": f"{p_label} vs. {n_label}".capitalize()
        }
        
    # --- SAVE RESULTS ---
    output_path = os.path.join(ROOT_DIR, "data", "production", "semantic_sum_labels.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Done! Saved 10k-word sum labels to {output_path}")
    
    # Print sample
    print("\nSample Sum-Based Dimension Labels (10k Words):")
    for d in range(10):
        print(f"Dim {d:3}: {output_data[str(d)]['dynamic_label']}")

if __name__ == "__main__":
    label_dimensions_with_word_sums()
