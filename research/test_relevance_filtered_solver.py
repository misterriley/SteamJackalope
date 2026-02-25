import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LassoCV

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, QUALITY_GRID_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, TAG_NORMS_FILE, 
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)

def run_filtered_test(user_id="76561198039155404"):
    print("--- Relevance-Filtered Solver Test ---")
    
    gt_path = "data/user_" + user_id + "_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    n_samples = len(y)
    
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z'])
    appid_to_idx = {aid: i for i, aid in enumerate(df_meta['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    
    W_TAG = 1.0
    W_SEM = 2.0
    W_TOP = 26.5
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices]
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    tag_feat = (tag_vectors / (tag_norms + DOT_PRODUCT_LAMBDA)) * W_TAG
    
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices]
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    sem_feat = (sem_vectors / (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * W_SEM
    
    topic_feat = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32) * W_TOP

    all_thematic = np.hstack([tag_feat, sem_feat, topic_feat])
    group_names = (['Tag'] * tag_feat.shape[1] + 
                   ['Semantic'] * sem_feat.shape[1] + 
                   ['Topic'] * topic_feat.shape[1])

    correlations = []
    for i in range(all_thematic.shape[1]):
        feat = all_thematic[:, i]
        corr = np.corrcoef(feat, y)[0, 1] if np.std(feat) > 1e-9 else 0.0
        correlations.append(abs(corr))
    
    allowed = n_samples - 7
    top_indices = np.argsort(-np.array(correlations))[:allowed]
    filtered_thematic = all_thematic[:, top_indices]
    
    q_feat = np.clip(np.load(QUALITY_GRID_FILE, mmap_mode='r')[10][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    meta_feat = np.clip(df_meta.iloc[user_indices][['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    X = np.hstack([q_feat.reshape(-1, 1), meta_feat, filtered_thematic])
    
    print("X Shape: " + str(X.shape) + " (Thematic Survivors: " + str(filtered_thematic.shape[1]) + ")")

    model = LassoCV(cv=5, max_iter=20000)
    model.fit(X, y)
    
    coeffs = model.coef_
    r2 = model.score(X, y)
    
    print("\n--- Solver Results ---")
    print("Model Training R^2: " + f"{r2:.4f}")
    print("Optimal Alpha:     " + f"{model.alpha_:.4f}")
    
    meta_weights = coeffs[:6]
    thematic_weights = coeffs[6:]
    
    survivor_groups = [group_names[i] for i in top_indices]
    tag_norm = np.linalg.norm([thematic_weights[i] for i, g in enumerate(survivor_groups) if g == 'Tag'])
    sem_norm = np.linalg.norm([thematic_weights[i] for i, g in enumerate(survivor_groups) if g == 'Semantic'])
    top_norm = np.linalg.norm([thematic_weights[i] for i, g in enumerate(survivor_groups) if g == 'Topic'])
    
    print("\n--- Slider Weight Equivalents ---")
    print("Quality:     " + f"{meta_weights[0]:.4f}")
    print("Metadata:    " + f"{np.linalg.norm(meta_weights[1:]):.4f}")
    print("Tag Match:   " + f"{tag_norm:.4f}")
    print("Semantic:    " + f"{sem_norm:.4f}")
    print("Topic Match: " + f"{top_norm:.4f}")

if __name__ == "__main__":
    run_filtered_test()
