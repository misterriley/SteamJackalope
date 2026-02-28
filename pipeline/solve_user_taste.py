import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
import os
import sys
import json
import ast
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    PRODUCTION_DATA_DIR,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    TAG_NAMES_FILE,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR,
    TOPIC_DISTRIBUTIONS_FILE,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX,
    DIFFICULTY_NEUTRAL_FALLBACK,
    TOPIC_GLOBAL_SCALING_FACTOR
)
from common.utils import calculate_jackalope_kernel, softmin_blend, calculate_linear_scores

def solve_user_taste(ground_truth_path, output_path=None):
    """
    Solves for user preference weights using the RESTORED model with GLOBAL scaling parity.
    """
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    sl_path = ground_truth_path.replace('_ground_truth.csv', '_soft_labels.csv')
    
    steam_library_appids = set()
    library_details = {}
    if os.path.exists(sl_path):
        df_sl = pd.read_csv(sl_path)
        steam_library_appids.update(df_sl['appid'].unique().tolist())
        for _, row in df_sl.iterrows():
            aid = int(row['appid'])
            library_details[aid] = {'playtime': float(row['playtime_forever']), 'p_plus_t': float(row['p_plus_t'])}
    
    discovery_exclude_appids = steam_library_appids.copy()
    if 'status' in df_gt.columns:
        discovery_exclude_appids.update(df_gt[df_gt['status'].isin(['backlog', 'played', 'rated'])]['appid'].tolist())
        discovery_exclude_appids.update(df_gt[df_gt['status'] == 'ignored']['appid'].tolist())
    
    df = df_gt[df_gt['status'] == 'rated'].dropna(subset=['actual_rating']).copy()
    user_appids = df['appid'].values
    y = df['actual_rating'].values
    
    print(f"Loading metadata and vectors...")
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    
    valid_mask = [aid in appid_to_idx for aid in user_appids]
    user_appids = user_appids[valid_mask]
    y = y[valid_mask]
    user_indices = [appid_to_idx[aid] for aid in user_appids]
    
    user_meta_df = full_metadata.iloc[user_indices].copy()
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)
    
    library_tags = user_meta_df['tags'].values
    
    # 3. Calculate 0.60 Kernel Matrix (NxN) with GLOBAL scaling
    N = len(user_indices)
    K = np.zeros((N, N))
    STRICT_ANCHORS = ["Platformer", "Puzzle", "Strategy", "RPG", "Roguelike", "Souls-like", "Metroidvania", "Action-Adventure", "Adventure"]
    
    print(f"Calculating {N}x{N} kernel matrix with GLOBAL scaling factors (T={TAG_GLOBAL_SCALING_FACTOR:.2f}, S={SEMANTIC_GLOBAL_SCALING_FACTOR:.2f})...")
    for i in range(N):
        t_sims = (np.dot(tag_vectors, tag_vectors[i]) / ((tag_norms + DOT_PRODUCT_LAMBDA) * (tag_norms[i] + DOT_PRODUCT_LAMBDA))) * TAG_GLOBAL_SCALING_FACTOR
        s_sims = (np.dot(sem_vectors, sem_vectors[i]) / ((sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA) * (sem_norms[i] + SEMANTIC_DOT_PRODUCT_LAMBDA))) * SEMANTIC_GLOBAL_SCALING_FACTOR
        
        zi = (topic_dist[i] - t_means) / (t_stds + 1e-9)
        zi[zi < 2.5] = 0
        ni = np.linalg.norm(zi) + 1e-9
        zj = (topic_dist - t_means) / (t_stds + 1e-9)
        zj[zj < 2.5] = 0
        nj = np.linalg.norm(zj, axis=1) + 1e-9
        top_sims = np.dot(zj, zi) / (nj * ni)
        
        tags_i = library_tags[i]
        if isinstance(tags_i, str): tags_i = ast.literal_eval(tags_i)
        max_i = max(tags_i.values()) if tags_i else 1.0
        anchors_i = [a for a in STRICT_ANCHORS if tags_i.get(a, 0) / max_i > 0.25]

        for j in range(N):
            consensus = softmin_blend([float(t_sims[j]), float(s_sims[j]), float(top_sims[j] * 0.1)], temperature=0.01)
            pure = (t_sims[j] * 0.25 + s_sims[j] * 0.25 + consensus * 0.5)
            
            tags_j = library_tags[j]
            if isinstance(tags_j, str): tags_j = ast.literal_eval(tags_j)
            for a in anchors_i:
                if a not in tags_j:
                    pure *= 0.001
                    break
            K[j, i] = pure

    # 4. Regression (No local scaling)
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_all = [np.corrcoef(np.clip(quality_grid[k][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX), y)[0, 1] if np.std(quality_grid[k][user_indices]) > 1e-9 else 0 for k in range(quality_grid.shape[0])]
    best_q_idx = np.argmax(np.abs(q_all))
    q_global = np.clip(quality_grid[best_q_idx][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
    X_meta = np.clip(user_meta_df[meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    X = np.hstack([q_global.reshape(-1, 1), X_meta, K])
    
    print(f"Solving Hybrid Ridge (GLOBAL PARITY) for {N} samples...")
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0]).fit(X, y)
    r2_train = model.score(X, y)
    print(f"Model Training R^2: {r2_train:.4f} (Alpha: {model.alpha_:.2f})")

    # --- RESULT GENERATION ---
    q_coeff = model.coef_[0]
    meta_coeffs = model.coef_[1:6]
    kernel_coeffs = model.coef_[6:]
    
    sorted_anchor_idxs = sorted(range(len(kernel_coeffs)), key=lambda idx: abs(kernel_coeffs[idx]), reverse=True)
    active_anchors = [{'appid': int(user_appids[idx]), 'name': str(user_meta_df.iloc[idx]['name']), 'weight': float(kernel_coeffs[idx])} for idx in sorted_anchor_idxs if abs(kernel_coeffs[idx]) > 1e-5]
    
    tag_match_weight = np.mean(np.abs(kernel_coeffs)) if len(kernel_coeffs) > 0 else 0.0
    
    # Representative preview vectors
    if active_anchors:
        best_anchor_idx = appid_to_idx[active_anchors[0]['appid']]
        vibe_unit = np.load(TAG_VECTORS_FILE, mmap_mode='r')[best_anchor_idx] / (np.load(TAG_NORMS_FILE, mmap_mode='r')[best_anchor_idx] + 1e-9)
        sem_unit = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[best_anchor_idx] / (np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[best_anchor_idx] + 1e-9)
        top_coeffs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[best_anchor_idx]
    else: vibe_unit, sem_unit, top_coeffs = np.zeros(231), np.zeros(235), np.zeros(250)

    result = {
        'metadata': {'quality': float(q_coeff), 'age': float(meta_coeffs[0]), 'popularity': float(meta_coeffs[1]), 'length': float(meta_coeffs[2]), 'difficulty': float(meta_coeffs[3]), 'price': float(meta_coeffs[4]), 'tag_match': float(tag_match_weight), 'semantic': float(tag_match_weight), 'topic_match': float(tag_match_weight * 0.1)},
        'kernel_anchors': active_anchors[:50], 'r2': float(r2_train), 
        'vibe_vector': vibe_unit.tolist(), 'semantic_vibe_vector': sem_unit.tolist(), 'topic_vibe_vector': top_coeffs.tolist(),
        'intercept': float(model.intercept_), 'scaling_factor': 3.0, 'library_appids': [int(aid) for aid in discovery_exclude_appids],
        'rated_appids': [int(aid) for aid in user_appids], 'library_size': len(discovery_exclude_appids)
    }
    
    if output_path:
        with open(output_path, 'w') as f: json.dump(result, f, indent=4)
    return result

if __name__ == "__main__":
    sid = sys.argv[1]
    solve_user_taste(f"data/user_{sid}_ground_truth.csv", output_path=f"data/user_{sid}_taste_profile.json")
