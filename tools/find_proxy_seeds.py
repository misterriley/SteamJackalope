import pandas as pd
import numpy as np
import os
import json
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    W_TAG_FILE,
    METADATA_FILE,
    ROOT_DIR,
    DOT_PRODUCT_LAMBDA
)

def find_best_regression_proxy(taste_profile_path, top_k=10):
    """
    Finds the games whose tag vectors most closely match the solved regression coefficients.
    """
    # 1. Load Taste Profile
    print(f"Loading taste profile from {taste_profile_path}...")
    with open(taste_profile_path, 'r') as f:
        profile = json.load(f)
    
    beta_white = np.array(profile['vibe_vector'], dtype=np.float32)
    
    # 2. Scale beta_white to the mean dataset norm (found to be 4.52)
    target_norm = 4.5195 
    current_norm = np.linalg.norm(beta_white)
    scaled_beta = beta_white * (target_norm / current_norm)
    
    # 3. Load Production Tag Vectors (Whitened)
    all_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    
    # 4. Calculate alignment (Dot Product)
    print("Calculating alignment with regression gradient...")
    dots = np.dot(all_vectors.astype(np.float32), scaled_beta)
    
    from common.constants import TAG_NORMS_FILE
    dataset_norms = np.load(TAG_NORMS_FILE).astype(np.float32)
    scores = dots / ((dataset_norms * target_norm) + DOT_PRODUCT_LAMBDA)
    
    # 5. Fetch Metadata
    metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'tags'])
    top_indices = np.argsort(-scores)[:top_k]
    
    print("\n" + "="*60)
    print("BEST PROXY GAMES (Representative Seeds)")
    print("These games' tag profiles act like your personal regression model.")
    print("="*60)
    
    for i, idx in enumerate(top_indices):
        name = metadata.iloc[idx]['name']
        score = scores[idx]
        tags = metadata.iloc[idx]['tags']
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        print(f"{i+1}. {safe_name:30} | Score: {score:.4f}")
        print(f"   Tags: {tags[:100]}...")
        print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/find_proxy_seeds.py <taste_profile_json>")
        sys.exit(1)
        
    profile_path = sys.argv[1]
    find_best_regression_proxy(profile_path)
