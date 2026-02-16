import pandas as pd
import numpy as np
import os
import json
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    TAG_NORMS_FILE,
    METADATA_FILE,
    DOT_PRODUCT_LAMBDA,
    ROOT_DIR
)

def search_perfect_game(taste_profile_path, top_k=20):
    """
    Uses the solved taste vector to find the closest matches in the database.
    Scales the taste vector to the mean dataset norm to ensure fair similarity scoring.
    """
    # 1. Load Taste Profile
    print(f"Loading taste profile from {taste_profile_path}...")
    with open(taste_profile_path, 'r') as f:
        profile = json.load(f)
    
    # The 'vibe_vector' is the beta solution in 128-dim whitened space
    taste_vector = np.array(profile['vibe_vector'], dtype=np.float32)
    
    # 2. Scale the Taste Vector
    dataset_norms = np.load(TAG_NORMS_FILE).astype(np.float64)
    target_norm = np.mean(dataset_norms)
    
    current_norm = np.linalg.norm(taste_vector)
    if current_norm > 1e-9:
        scaled_taste_vector = taste_vector * (target_norm / current_norm)
    else:
        scaled_taste_vector = taste_vector
        
    print(f"Taste vector scaled to target norm: {target_norm:.4f}")

    # 3. Load Production Tag Vectors
    print(f"Loading {TAG_VECTORS_FILE}...")
    all_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    
    # 4. Calculate Penalized Cosine Similarity
    print(f"Calculating similarity (Lambda={DOT_PRODUCT_LAMBDA:.4f})...")
    dots = np.dot(all_vectors.astype(np.float32), scaled_taste_vector)
    denominator = (dataset_norms * target_norm) + DOT_PRODUCT_LAMBDA
    similarities = dots / denominator
    
    # 5. Get Metadata for Names
    print(f"Loading metadata from {METADATA_FILE}...")
    metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    
    # 6. Rank Results
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'appid': int(metadata.iloc[idx]['appid']),
            'name': metadata.iloc[idx]['name'],
            'score': float(similarities[idx])
        })
        
    print("\n" + "="*60)
    print(f"TOP {top_k} MATCHES FOR YOUR TASTE PROFILE")
    print("="*60)
    for res in results:
        # Avoid ASCII issues by replacing chars
        safe_name = res['name'].encode('ascii', 'replace').decode('ascii')
        print(f"{res['appid']:<10} | {safe_name:35} | {res['score']:.4f}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/search_by_taste.py <taste_profile_json>")
        sys.exit(1)
        
    profile_path = sys.argv[1]
    search_perfect_game(profile_path)
