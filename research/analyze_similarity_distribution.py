import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, EMBEDDINGS_DESC_FILE, 
    TAG_NORMS_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR, SEMANTIC_GLOBAL_SCALING_FACTOR, METADATA_FILE
)

def analyze_distributions():
    print("Loading artifacts for similarity study...")
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')

    appid_to_idx = {aid: i for i, aid in enumerate(df_meta['appid'])}
    
    # Anchor Games
    anchors = [
        {"name": "Amnesia: The Dark Descent", "appid": 57300},
        {"name": "Portal 2", "appid": 620},
        {"name": "Stardew Valley", "appid": 413150},
        {"name": "Counter-Strike: Global Offensive", "appid": 730}
    ]

    for anchor in anchors:
        print("\n--- Analyzing Similarity Distribution for: " + anchor['name'] + " ---")
        idx = appid_to_idx[anchor['appid']]
        
        v_tag = tag_vectors[idx].astype(np.float32)
        n_tag = tag_norms[idx]
        tag_sims = (np.dot(tag_vectors.astype(np.float32), v_tag) / ((n_tag + DOT_PRODUCT_LAMBDA) * (tag_norms + DOT_PRODUCT_LAMBDA))) * TAG_GLOBAL_SCALING_FACTOR
        
        v_sem = sem_vectors[idx].astype(np.float32)
        sem_sims = (np.dot(sem_vectors.astype(np.float32), v_sem) / (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
        
        total_sim = tag_sims + sem_sims
        
        print("  Max:    " + f"{np.max(total_sim):.4f}")
        print("  99.9th: " + f"{np.percentile(total_sim, 99.9):.4f}" + " (~150 games)")
        print("  99th:   " + f"{np.percentile(total_sim, 99.0):.4f}" + " (~1500 games)")
        print("  95th:   " + f"{np.percentile(total_sim, 95.0):.4f}" + " (~7500 games)")
        print("  Mean:   " + f"{np.mean(total_sim):.4f}")
        print("  Std:    " + f"{np.std(total_sim):.4f}")

        top_idx = np.argsort(-total_sim)[1:6]
        print("  Top Neighbors:")
        for t_idx in top_idx:
            print("    - " + f"{total_sim[t_idx]:.4f}" + " | " + str(df_meta.iloc[t_idx]['name']))

if __name__ == "__main__":
    analyze_distributions()
