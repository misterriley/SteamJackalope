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
from common.utils import softmin_blend

def debug_similarities(seed_appid, target_appids):
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    
    # Load data
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy"))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy"))

    s_idx = appid_to_idx[seed_appid]
    
    print(f"Seed: {full_metadata.iloc[s_idx]['name']} ({seed_appid})")
    print("-" * 50)

    for target_appid in target_appids:
        t_idx = appid_to_idx[target_appid]
        
        # Tags
        v_s, v_t = tag_vectors[s_idx].astype(np.float32), tag_vectors[t_idx].astype(np.float32)
        n_s, n_t = tag_norms[s_idx], tag_norms[t_idx]
        tag_sim = (np.dot(v_s, v_t) / ((n_s + DOT_PRODUCT_LAMBDA) * (n_t + DOT_PRODUCT_LAMBDA))) * TAG_GLOBAL_SCALING_FACTOR
        
        # Semantics
        s_s, s_t = sem_vectors[s_idx].astype(np.float32), sem_vectors[t_idx].astype(np.float32)
        sn_s, sn_t = sem_norms[s_idx], sem_norms[t_idx]
        sem_sim = (np.dot(s_s, s_t) / ((sn_s + SEMANTIC_DOT_PRODUCT_LAMBDA) * (sn_t + SEMANTIC_DOT_PRODUCT_LAMBDA))) * SEMANTIC_GLOBAL_SCALING_FACTOR
        
        # Topics
        p_s = topic_dist[s_idx].astype(np.float32)
        p_t = topic_dist[t_idx].astype(np.float32)
        z_s = (p_s - t_means) / (t_stds + 1e-9)
        z_t = (p_t - t_means) / (t_stds + 1e-9)
        z_s[z_s < 1.5] = 0
        z_t[z_t < 1.5] = 0
        topic_sim = np.dot(z_s / (np.linalg.norm(z_s) + 1e-9), z_t / (np.linalg.norm(z_t) + 1e-9))
        topic_sim_scaled = topic_sim * 0.1
        
        consensus = softmin_blend([np.array([tag_sim]), np.array([sem_sim]), np.array([topic_sim_scaled])], temperature=0.01)[0]
        
        print(f"Target: {full_metadata.iloc[t_idx]['name']} ({target_appid})")
        print(f"  Tags: {tag_sim:.4f}")
        print(f"  Sem:  {sem_sim:.4f}")
        print(f"  Top:  {topic_sim_scaled:.4f}")
        print(f"  Consensus: {consensus:.4f}")
        print("-" * 20)

if __name__ == "__main__":
    # Ori vs Guacamelee 2, Wizardry 8, Avernum 4
    debug_similarities(1057090, [534550, 245450, 206020])
