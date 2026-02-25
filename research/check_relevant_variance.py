import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, 
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, TAG_NORMS_FILE, PRODUCTION_DATA_DIR,
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR, SEMANTIC_GLOBAL_SCALING_FACTOR, TOPIC_GLOBAL_SCALING_FACTOR
)

def check_relevant_variance(user_id="76561198039155404"):
    # 1. Load User Data
    gt_path = "data/user_" + user_id + "_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    n_samples = len(y)
    
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid'])
    appid_to_idx = {aid: i for i, aid in enumerate(df_meta['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    
    # 2. Extract Features
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices]
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    tag_feat = (tag_vectors / (tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices]
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    sem_feat = (sem_vectors / (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
    
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy"))
    topic_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy"))
    topic_feat = ((topic_dist - topic_means) / (topic_stds + 1e-10)) * np.sqrt(0.5) * TOPIC_GLOBAL_SCALING_FACTOR

    all_thematic = np.hstack([tag_feat, sem_feat, topic_feat])
    group_names = (['Tag'] * tag_feat.shape[1] + 
                   ['Semantic'] * sem_feat.shape[1] + 
                   ['Topic'] * topic_feat.shape[1])

    # 3. Find Survivors
    correlations = []
    for i in range(all_thematic.shape[1]):
        feat = all_thematic[:, i]
        corr = np.corrcoef(feat, y)[0, 1] if np.std(feat) > 1e-9 else 0.0
        correlations.append(abs(corr))
    
    allowed = n_samples - 6
    survivor_indices = np.argsort(-np.array(correlations))[:allowed]
    survivor_X = all_thematic[:, survivor_indices]
    survivor_groups = [group_names[i] for i in survivor_indices]

    # 4. Analyze Variance
    variances = np.var(survivor_X, axis=0)
    
    print("Analyzing " + str(len(survivor_indices)) + " survivors...")
    print("\n--- Variance Stats for Filtered Features ---")
    print("Mean Variance: " + f"{np.mean(variances):.4f}")
    print("Std Variance:  " + f"{np.std(variances):.4f}")
    print("Min Variance:  " + f"{np.min(variances):.4f}")
    print("Max Variance:  " + f"{np.max(variances):.4f}")

    for g in ['Tag', 'Semantic', 'Topic']:
        g_vars = [variances[i] for i, group in enumerate(survivor_groups) if group == g]
        if g_vars:
            print(f"{g:<10} Mean Var: {np.mean(g_vars):.4f} (count: {len(g_vars)})")

if __name__ == "__main__":
    check_relevant_variance()
