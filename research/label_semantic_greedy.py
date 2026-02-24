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

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    MODEL_NAME,
    W_DESC_FILE,
    ROOT_DIR,
    METADATA_FILE
)

def label_dimensions_greedy():
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
    # Filter out common stop words or useless words if needed, but 10k top words should be fine
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    word_counts = Counter(words)
    
    # Take the top 10,000 unique single words
    vocab = [w for w, c in word_counts.most_common(10000)]
    print(f"Vocabulary size: {len(vocab)} unique single words")
    
    # --- ENCODE AND PROJECT ---
    print("Encoding vocabulary...")
    embeddings_raw = model.encode(vocab, show_progress_bar=True, batch_size=128)
    
    print("Projecting into whitened space...")
    V = np.dot(embeddings_raw, W).astype(np.float32) # (VocabSize, Dim)
    
    # --- GREEDY BAG-OF-WORDS APPROXIMATION ---
    print("Greedily building bag-of-words for each dimension...")
    
    output_data = {}
    MAX_WORDS = 3
    
    # Precompute norms squared for cosine sim efficiency
    v_norms_sq = np.sum(V**2, axis=1)
    
    for d in tqdm(range(num_dims), desc="Labeling Dimensions"):
        target_pos = np.zeros(num_dims, dtype=np.float32)
        target_pos[d] = 1.0
        
        target_neg = np.zeros(num_dims, dtype=np.float32)
        target_neg[d] = -1.0
        
        def build_greedy_bag(target):
            bag_indices = []
            bag_vec = np.zeros(num_dims, dtype=np.float32)
            current_best_sim = -1.0
            
            for _ in range(MAX_WORDS):
                # We want to maximize cosine_sim(bag_vec + V[i], target)
                # Cosine sim = (bag_vec + V[i]).dot(target) / ||bag_vec + V[i]||
                # Numerator: bag_vec.dot(target) + V[i, d] * target[d]
                
                # Since target is e_d or -e_d, dot product is just the d-th component
                # multiplied by the target's sign.
                target_val = target[d]
                
                num = bag_vec[d] * target_val + V[:, d] * target_val
                
                # Denominator: sqrt(||bag_vec||^2 + ||V[i]||^2 + 2 * bag_vec.dot(V[i]))
                bag_norm_sq = np.sum(bag_vec**2)
                dots = np.dot(V, bag_vec)
                den = np.sqrt(np.maximum(bag_norm_sq + v_norms_sq + 2 * dots, 1e-9))
                
                sims = num / den
                
                # Mask out already selected words to avoid repetition
                for idx in bag_indices:
                    sims[idx] = -1.0
                
                best_idx = np.argmax(sims)
                if sims[best_idx] > current_best_sim + 0.001: # Small epsilon for improvement
                    current_best_sim = sims[best_idx]
                    bag_indices.append(int(best_idx))
                    bag_vec += V[best_idx]
                else:
                    break
            
            return [vocab[i] for i in bag_indices]

        pos_bag = build_greedy_bag(target_pos)
        neg_bag = build_greedy_bag(target_neg)
        
        p_label = " + ".join(pos_bag)
        n_label = " + ".join(neg_bag)
        
        output_data[str(d)] = {
            "positive_label": p_label,
            "negative_label": n_label,
            "dynamic_label": f"{p_label} vs. {n_label}".capitalize()
        }
        
    # --- SAVE RESULTS ---
    output_path = os.path.join(ROOT_DIR, "data", "production", "semantic_greedy_labels.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    # Also update the primary semantic labels if this is the new standard
    # The app usually looks for semantic_sum_labels.json or similar.
    # I'll save it to both for safety or check where it's used.
    
    print(f"Done! Saved greedy labels to {output_path}")
    
    # Print sample
    print("\nSample Greedy Dimension Labels (10k Words):")
    for d in range(10):
        print(f"Dim {d:3}: {output_data[str(d)]['dynamic_label']}")

if __name__ == "__main__":
    label_dimensions_greedy()
