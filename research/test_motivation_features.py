import pandas as pd
import numpy as np
import os
import sys
import ast
import json
import re
from sklearn.linear_model import LassoCV
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

def research_motivation_features(steamid="76561198039155404"):
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
    
    # 2. Build Internal Kernel Matrix
    N = len(user_indices)
    K = np.zeros((N, N))
    STRICT_ANCHORS = ["Platformer", "Puzzle", "Strategy", "RPG", "Roguelike", "Souls-like", "Metroidvania", "Action-Adventure", "Adventure"]
    anchor_masks_rated = {a: full_tags_series.iloc[user_indices].str.contains(f"'{a}':", na=False).values for a in STRICT_ANCHORS}

    print(f"Building Kernel Matrix for {N} games...")
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

    # 3. Calculate Motivation Loadings
    motivation_library_path = os.path.join(PRODUCTION_DATA_DIR, "motivations_library.json")
    mot_loadings = np.zeros((N, 12))
    mot_names = []
    
    if os.path.exists(motivation_library_path):
        print("Calculating Motivation Loadings...")
        library = json.load(open(motivation_library_path, 'r'))
        for m_idx, (mot_name, v) in enumerate(library.items()):
            mot_names.append(mot_name)
            sem_m = np.array(v.get('semantic_vector', v.get('semantic'))).astype(np.float32).flatten()
            top_m = np.array(v.get('topic_vector', v.get('topic'))).astype(np.float32).flatten()
            tag_m = np.array(v.get('tag_vector', [])).astype(np.float32).flatten()[:tag_vectors.shape[1]]
            
            r_tag = np.dot(tag_vectors[user_indices], tag_m) if len(tag_m) > 0 else np.zeros(N)
            r_sem = np.dot(sem_vectors[user_indices], sem_m) / (sem_norms[user_indices] + 1e-9)
            tz_m = (top_m - t_means) / (t_stds + 1e-9); tz_m[tz_m < 0] = 0; tn_m = np.linalg.norm(tz_m) + 1e-9
            uz_batch = (topic_dist_all[user_indices] - t_means) / (t_stds + 1e-9); uz_batch[uz_batch < 0] = 0; un_batch = np.linalg.norm(uz_batch, axis=1, keepdims=True) + 1e-9
            r_top = np.dot(uz_batch / un_batch, tz_m / tn_m)
            
            mot_loadings[:, m_idx] = (r_tag * 0.2 + r_sem * 0.4 + r_top * 0.4)

    # 4. Prepare Metadata
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
    X_meta = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    q_best = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')[0]
    X_meta = np.hstack([np.clip(q_best[user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX).reshape(-1, 1), X_meta])

    # 5. Comparative Regression
    X_baseline = np.hstack([X_meta, K])
    X_extended = np.hstack([X_meta, mot_loadings, K])
    
    print("\n--- Comparative Predictive Power ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_base, scores_ext = [], []
    
    for train_idx, test_idx in kf.split(X_baseline):
        m_base = LassoCV(cv=3, max_iter=10000).fit(X_baseline[train_idx], y[train_idx])
        scores_base.append(m_base.score(X_baseline[test_idx], y[test_idx]))
        m_ext = LassoCV(cv=3, max_iter=10000).fit(X_extended[train_idx], y[train_idx])
        scores_ext.append(m_ext.score(X_extended[test_idx], y[test_idx]))
        
    print(f"Baseline (Meta + Kernel) CV R^2: {np.mean(scores_base):.4f}")
    print(f"Extended (Meta + Mots + Kernel) CV R^2: {np.mean(scores_ext):.4f}")
    
    final_model = LassoCV(cv=5).fit(X_extended, y)
    mot_coeffs = final_model.coef_[6:18]
    print("\n--- Motivation Feature Importance ---")
    for name, coeff in zip(mot_names, mot_coeffs):
        if abs(coeff) > 1e-5:
            print(f"  - {name}: {coeff:+.4f}")

if __name__ == "__main__":
    research_motivation_features()
