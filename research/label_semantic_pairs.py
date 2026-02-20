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
    METADATA_FILE,
    TAG_NAMES_FILE
)

def label_dimensions_with_pairs():
    print("Loading model and whitening matrix...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    if not os.path.exists(W_DESC_FILE):
        print(f"Error: {W_DESC_FILE} not found.")
        return
        
    W = np.load(W_DESC_FILE).astype(np.float32)
    num_dims = W.shape[1]
    
    # --- GATHER VOCABULARY ---
    print("Gathering vocabulary...")
    words = set()
    
    # 1. Tags (High priority)
    if os.path.exists(TAG_NAMES_FILE):
        with open(TAG_NAMES_FILE, 'r') as f:
            tags = json.load(f)
            words.update([t.lower() for t in tags])

    # 2. Common Adjectives
    adj_path = os.path.join(ROOT_DIR, "common", "common_adjectives.txt")
    if os.path.exists(adj_path):
        with open(adj_path, 'r') as f:
            words.update([l.strip().lower() for l in f if l.strip()])
            
    # 3. Top words from descriptions
    if os.path.exists(METADATA_FILE):
        df = pd.read_parquet(METADATA_FILE, columns=['short_description'])
        all_text = " ".join(df['short_description'].fillna('').tolist()).lower()
        extracted = re.findall(r'\b[a-z]{3,}\b', all_text)
        counts = Counter(extracted)
        # Take top 3000 frequent words to keep complexity manageable
        for w, c in counts.most_common(3000):
            words.add(w)
        
    vocab = sorted(list(words))
    # Limit total vocab size to prevent OOM/Time issues if too large
    if len(vocab) > 4000:
        import random
        random.seed(42)
        vocab = random.sample(vocab, 4000)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # --- ENCODE AND PROJECT ---
    print("Encoding vocabulary...")
    embeddings_raw = model.encode(vocab, show_progress_bar=True, batch_size=128)
    
    print("Projecting into whitened space...")
    V = np.dot(embeddings_raw, W) # (N, D)
    
    # --- FIND ALIGNED PAIRS ---
    print("Finding aligned pairs for each dimension...")
    
    # Storage for best pair per dimension
    # Format: (score, word_i, word_j)
    # We initialize with a safe dummy pair
    best_pairs = [(0.0, "word_a", "word_b") for _ in range(num_dims)]
    
    # We process in batches of row_i to vectorize the comparison against all row_j
    batch_size = 50
    N = len(vocab)
    D = num_dims
    
    # Pre-convert to float32 for speed/precision balance
    V = V.astype(np.float32)
    
    for start_i in tqdm(range(0, N, batch_size), desc="Scanning pairs"):
        end_i = min(start_i + batch_size, N)
        
        # Batch of vectors A: (batch, D)
        A = V[start_i:end_i]
        
        # We want difference with all vectors B: (N, D)
        # Broadcasting A - B results in (batch, N, D)
        
        # Expand dims for broadcast
        # A: (batch, 1, D)
        # V: (1, N, D)
        Diff = A[:, np.newaxis, :] - V[np.newaxis, :, :] # (batch, N, D)
        
        # Compute norms of difference vectors
        # Norms: (batch, N)
        Norms = np.linalg.norm(Diff, axis=2)
        
        # Avoid division by zero (self-comparison)
        Norms[Norms < 1e-6] = 1.0 
        
        # For each dimension d, we want (Diff[:, :, d] / Norms)
        for d in range(D):
            # Projections on axis d
            # Proj: (batch, N)
            Proj = Diff[:, :, d]
            
            # Scores: (batch, N) - Cosine similarity of difference vector to axis d
            Scores = Proj / Norms
            
            AbsScores = np.abs(Scores)
            
            # Find max absolute score in this batch
            # We can flatten to find the single max
            flat_idx = np.argmax(AbsScores)
            batch_max = AbsScores.flat[flat_idx]
            
            if batch_max > best_pairs[d][0]:
                # Found a new best
                r, c = np.unravel_index(flat_idx, AbsScores.shape)
                actual_score = Scores[r, c]
                
                word_a = vocab[start_i + r]
                word_b = vocab[c]
                
                # Order words so the difference is Positive along axis
                if actual_score < 0:
                    best_pairs[d] = (abs(actual_score), word_b, word_a)
                else:
                    best_pairs[d] = (abs(actual_score), word_a, word_b)
                    
    # --- SAVE RESULTS ---
    print("Saving results...")
    output_data = {}
    for i, (score, w1, w2) in enumerate(best_pairs):
        output_data[str(i)] = {
            "positive_word": w1,
            "negative_word": w2,
            "alignment_score": float(score),
            "pair_label": f"{w1} vs. {w2}"
        }
        
    output_path = os.path.join(ROOT_DIR, "data", "production", "semantic_pair_labels.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Done! Saved pair labels to {output_path}")
    
    # Print sample
    print("\nTop Dimension Pairs:")
    # Sort dimensions by alignment score to see the most "meaningful" axes
    sorted_dims = sorted(range(num_dims), key=lambda x: best_pairs[x][0], reverse=True)
    
    for i in sorted_dims[:20]:
        score, w1, w2 = best_pairs[i]
        print(f"Dim {i:3}: {w1:15} vs {w2:15} (Align: {score:.4f})")

if __name__ == "__main__":
    label_dimensions_with_pairs()
