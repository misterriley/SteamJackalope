import pandas as pd
import numpy as np
import os
import sys
import ast
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, PRODUCTION_DATA_DIR, TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA, TAG_GLOBAL_SCALING_FACTOR, EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE, SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, TOPIC_DISTRIBUTIONS_FILE,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import calculate_jackalope_kernel

def compare_l1_l2(steamid="76561198039155404"):
    # 1. Load Data
    df_gt = pd.read_csv(f"data/user_{steamid}_ground_truth.csv")
    df_rated = df_gt[df_gt['status'] == 'rated'].dropna(subset=['actual_rating']).copy()
    y = df_rated['actual_rating'].values
    user_appids = df_rated['appid'].values
    
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'tags', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z', 'price_z'])
    appid_to_idx = {int(aid): i for i, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    y = y[[aid in appid_to_idx for aid in user_appids]]
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_dist_all = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)
    full_tags_series = full_metadata['tags'].fillna('').astype(str)
    
    # Kernel constants
    STRICT_ANCHORS = ["Platformer", "Puzzle", "Strategy", "RPG", "Roguelike", "Souls-like", "Metroidvania", "Action-Adventure", "Adventure"]
    anchor_masks_rated = {a: full_tags_series.iloc[user_indices].str.contains(f"'{a}':", na=False).values for a in STRICT_ANCHORS}

    # 2. Build X_hybrid (Metadata + Kernel)
    N = len(user_indices)
    K = np.zeros((N, N))
    print(f"Building internal Kernel Matrix for {N} games...")
    for i in range(N):
        idx_i = user_indices[i]
        tags_i = ast.literal_eval(full_metadata.iloc[idx_i]['tags']) if isinstance(full_metadata.iloc[idx_i]['tags'], str) else full_metadata.iloc[idx_i]['tags']
        max_i = max(tags_i.values()) if tags_i else 1.0
        K[:, i] = calculate_jackalope_kernel(
            tag_vectors[user_indices], tag_norms[user_indices], tag_vectors[idx_i], tag_norms[idx_i],
            sem_vectors[user_indices], sem_norms[user_indices], sem_vectors[idx_i], sem_norms[idx_i],
            topic_dist_all[user_indices], topic_dist_all[idx_i],
            t_means, t_stds, TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA, SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA,
            seed_anchors=[a for a in STRICT_ANCHORS if tags_i.get(a, 0) / max_i > 0.25],
            candidate_anchor_masks=anchor_masks_rated,
            full_tags_series=full_tags_series.iloc[user_indices]
        )

    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
    X_meta = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    X_hybrid = np.hstack([X_meta, K])

    # 3. Solve and Compare
    print("\n--- Model Comparison ---")
    lasso = LassoCV(cv=5).fit(X_hybrid, y)
    l1_r2 = lasso.score(X_hybrid, y)
    l1_active = np.sum(abs(lasso.coef_) > 1e-5)
    
    ridge = RidgeCV(cv=5).fit(X_hybrid, y)
    l2_r2 = ridge.score(X_hybrid, y)
    l2_active = np.sum(abs(ridge.coef_) > 1e-5)
    
    print(f"Lasso (L1) Training R^2: {l1_r2:.4f} | Active Features: {l1_active}")
    print(f"Ridge (L2) Training R^2: {l2_r2:.4f} | Active Features: {l2_active}")
    
    # 4. Cross-Validation Test
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    l1_scores, l2_scores = [], []
    
    for train_idx, test_idx in kf.split(X_hybrid):
        X_train, X_test = X_hybrid[train_idx], X_hybrid[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m1 = LassoCV(cv=3, max_iter=10000).fit(X_train, y_train)
        m2 = RidgeCV(cv=3).fit(X_train, y_train)
        l1_scores.append(m1.score(X_test, y_test))
        l2_scores.append(m2.score(X_test, y_test))
        
    print(f"\nCV Validation R^2 (Generalization):")
    print(f"Lasso (L1): {np.mean(l1_scores):.4f}")
    print(f"Ridge (L2): {np.mean(l2_scores):.4f}")

if __name__ == "__main__":
    compare_l1_l2()
