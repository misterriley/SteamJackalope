import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, QUALITY_GRID_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA, TOPIC_DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR, SEMANTIC_GLOBAL_SCALING_FACTOR, TOPIC_GLOBAL_SCALING_FACTOR,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)

def check_input_variance(user_id="76561198039155404"):
    gt_path = "data/user_" + user_id + "_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid'])
    appid_to_idx = {aid: i for i, aid in enumerate(df_meta['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    
    print("Analyzing input variance for " + str(len(user_indices)) + " games...")

    q_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')[10][user_indices]
    q_feat = np.clip(q_grid, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices]
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    tag_feat = (tag_vectors / (tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices]
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    sem_feat = (sem_vectors / (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
    
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices]
    topic_feat = (topic_dist / (1.0 + TOPIC_DOT_PRODUCT_LAMBDA)) * TOPIC_GLOBAL_SCALING_FACTOR

    report = [
        ("Quality", np.var(q_feat)),
        ("Tags (Avg/Dim)", np.mean(np.var(tag_feat, axis=0))),
        ("Semantics (Avg/Dim)", np.mean(np.var(sem_feat, axis=0))),
        ("Topics (Avg/Dim)", np.mean(np.var(topic_feat, axis=0)))
    ]
    
    print("\n--- Average Feature Variance per Dimension ---")
    for name, var in report:
        print(name + ": " + f"{var:.6f}")

if __name__ == "__main__":
    check_input_variance()
