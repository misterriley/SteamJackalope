import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys
import json
import torch

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    MODEL_NAME,
    W_DESC_FILE,
    ROOT_DIR,
    TAG_NAMES_FILE,
    METADATA_FILE
)

def label_semantic_dimensions():
    print("Loading model and whitening matrix...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    if not os.path.exists(W_DESC_FILE):
        print(f"Error: {W_DESC_FILE} not found.")
        return
        
    W = np.load(W_DESC_FILE).astype(np.float32)
    num_dims = W.shape[1]
    print(f"Whitening matrix shape: {W.shape} (K={num_dims})")
    
    # --- GATHER WORD LIST ---
    print("Gathering word list...")
    words = set()
    
    # 1. Tags
    if os.path.exists(TAG_NAMES_FILE):
        with open(TAG_NAMES_FILE, 'r') as f:
            words.update(json.load(f))
            
    # 2. Common Adjectives
    adj_path = os.path.join(ROOT_DIR, "common", "common_adjectives.txt")
    if os.path.exists(adj_path):
        with open(adj_path, 'r') as f:
            words.update([l.strip() for l in f if l.strip()])
            
    # 3. Genres from metadata
    if os.path.exists(METADATA_FILE):
        df = pd.read_parquet(METADATA_FILE, columns=['genres'])
        for genres_str in df['genres'].dropna().unique():
            if genres_str.startswith('['):
                import ast
                try:
                    words.update(ast.literal_eval(genres_str))
                except: pass
            else:
                words.update([g.strip() for g in genres_str.split(',')])
                
    # 4. Hand-picked gameplay terms
    gameplay_terms = [
        "fast-paced", "permadeath", "crafting", "procedural", "pixel art", 
        "open world", "narrative-driven", "horror", "co-op", "difficult",
        "relaxing", "casual", "violent", "multiplayer", "singleplayer",
        "indie", "strategy", "tactical", "turn-based", "real-time",
        "first-person", "third-person", "isometric", "top-down",
        "2D", "3D", "retro", "modern", "stylized", "realistic",
        "fantasy", "sci-fi", "cyberpunk", "steampunk", "lovecraftian",
        "mystery", "romance", "drama", "comedy", "satire", "political",
        "educational", "puzzle", "physics-based", "platformer", "shooter",
        "rpg", "roguelike", "simulation", "management", "exploration",
        "base-building", "survival", "sandbox", "bullet-hell", "metroidvania",
        "soulslike", "card-game", "deckbuilder", "tower-defense", "visual-novel",
        "walking-simulator", "interactive-fiction", "dating-sim", "hentai",
        "anime", "manga", "cinematic", "story-rich", "choices-matter",
        "multiple-endings", "emotional", "atmospheric", "surreal", "dark",
        "gritty", "colorful", "minimalist", "hand-drawn", "low-poly",
        "vr", "non-vr", "controller-support", "moddable", "free-to-play"
    ]
    words.update(gameplay_terms)
    
    word_list = sorted(list(words))
    print(f"Total words to encode: {len(word_list)}")
    
    # --- ENCODE AND PROJECT ---
    print("Encoding words...")
    # SentenceTransformer.encode returns raw embeddings (768-dim for mpnet)
    embeddings_raw = model.encode(word_list, show_progress_bar=True, batch_size=128)
    
    print("Projecting into whitened space...")
    # Whitening: (v - mean) @ W. But solve_user_taste uses uncentered ZCA (mean=0)
    embeddings_whitened = np.dot(embeddings_raw, W)
    
    # Unit normalize projected word vectors to calculate cosine similarity to dimensions
    # Actually, dimension i is represented by the basis vector e_i in whitened space.
    # Cosine similarity of whitened word v to dimension i is v[i] / ||v||.
    norms = np.linalg.norm(embeddings_whitened, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    cosine_sims = embeddings_whitened / norms
    
    # --- LABEL DIMENSIONS ---
    print("Labeling dimensions...")
    dimension_labels = {}
    
    for i in range(num_dims):
        # Loading on dimension i
        loadings = cosine_sims[:, i]
        
        # Top positive
        pos_indices = np.argsort(-loadings)[:10]
        top_pos = [(word_list[idx], float(loadings[idx])) for idx in pos_indices if loadings[idx] > 0.05]
        
        # Top negative
        neg_indices = np.argsort(loadings)[:10]
        top_neg = [(word_list[idx], float(loadings[idx])) for idx in neg_indices if loadings[idx] < -0.05]
        
        dimension_labels[str(i)] = {
            "top_positive": top_pos,
            "top_negative": top_neg
        }
        
    # --- SAVE RESULTS ---
    output_path = os.path.join(ROOT_DIR, "data", "production", "semantic_dimension_labels.json")
    with open(output_path, 'w') as f:
        json.dump(dimension_labels, f, indent=4)
        
    print(f"Done! Saved labels to {output_path}")
    
    # Print a few examples
    print("\nSample Dimension Labels:")
    for i in range(min(5, num_dims)):
        dim_str = str(i)
        print(f"\nDimension {i}:")
        pos_words = ', '.join([w for w, s in dimension_labels[dim_str]['top_positive'][:5]])
        neg_words = ', '.join([w for w, s in dimension_labels[dim_str]['top_negative'][:5]])
        print(f"  Positive: {pos_words}")
        print(f"  Negative: {neg_words}")

if __name__ == "__main__":
    label_semantic_dimensions()
