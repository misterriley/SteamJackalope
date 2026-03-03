import pandas as pd
import numpy as np
import os
import sys
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, PRODUCTION_DATA_DIR,
    TAG_NORMS_FILE, DOT_PRODUCT_LAMBDA, TAG_GLOBAL_SCALING_FACTOR,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    SEMANTIC_DOT_PRODUCT_LAMBDA, SEMANTIC_GLOBAL_SCALING_FACTOR,
    TOPIC_DISTRIBUTIONS_FILE
)

def test_cosine(seed_appid, target_appids):
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')

    s_idx = appid_to_idx[seed_appid]
    v_s = tag_vectors[s_idx].astype(np.float32)
    n_s = tag_norms[s_idx]
    u_s = v_s / (n_s + 1e-9)
    
    print(f"Seed: {full_metadata.iloc[s_idx]['name']}")
    
    for t_id in target_appids:
        t_idx = appid_to_idx[t_id]
        v_t = tag_vectors[t_idx].astype(np.float32)
        n_t = tag_norms[t_idx]
        u_t = v_t / (n_t + 1e-9)
        
        cos_sim = np.dot(u_s, u_t)
        
        # Original penalized dot
        pen_sim = (np.dot(v_s, v_t) / ((n_s + 0.1) * (n_t + 0.1))) * 1.0 # scaling factor 1.0
        
        print(f"Target: {full_metadata.iloc[t_idx]['name']}")
        print(f"  Unit Cosine: {cos_sim:.4f}")
        print(f"  Penalized:   {pen_sim:.4f}")

if __name__ == "__main__":
    test_cosine(1057090, [367520, 534550, 245450, 206020])
