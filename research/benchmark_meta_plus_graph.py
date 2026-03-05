import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import os
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, MIGS, normalize_string

def benchmark_meta_plus_graph(user_id="76561198039155404"):
    gt_path = f"data/user_{user_id}_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    y_dev_global = y - 5.0
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices]
    N = len(user_indices)

    # --- STATIC METADATA FEATURES ---
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_feat = to_z(quality_grid[0][user_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z', 'tone_z']
    X_meta = np.zeros((N, len(meta_cols)))
    for j, col in enumerate(meta_cols):
        X_meta[:, j] = to_z(user_meta_df[col].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    
    mig_mask_array = np.zeros((len(full_metadata), len(MIGS)), dtype=bool)
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for j, (group, tags) in enumerate(MIGS.items()):
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            mig_mask_array[:, j] |= tag_series_full.str.contains(pattern, regex=True).values
    X_mig = mig_mask_array[user_indices].astype(float)
    X_static = np.hstack([q_feat.reshape(-1, 1), X_meta, X_mig])

    # --- BEHAVIORAL GRAPH DATA ---
    all_graph_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)
    user_graph_vectors = all_graph_vectors[user_indices]
    
    # NxN Graph Similarity Matrix
    dot_graph = np.dot(user_graph_vectors, user_graph_vectors.T)
    g_norms = np.linalg.norm(user_graph_vectors, axis=1)
    G_full = dot_graph / (g_norms[:, None] * g_norms[None, :] + 1e-9)
    G_full = np.maximum(0, G_full)

    # Leak-Proof Mask (Series/Twins)
    user_names = user_meta_df['name'].tolist()
    lp_mask = np.ones((N, N), dtype=bool)
    for i in range(N):
        clean_i = set(normalize_string(user_names[i]).split())
        if len(clean_i) < 2: continue 
        for j in range(i + 1, N):
            clean_j = set(normalize_string(user_names[j]).split())
            overlap = clean_i.intersection(clean_j)
            if len(overlap) >= 2 and (len(overlap) / max(len(clean_i), len(clean_j)) > 0.7):
                lp_mask[i, j] = lp_mask[j, i] = False

    # --- STRICT OOS LOOP ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oos_scores = []
    
    print(f"\n--- Metadata + Graph Ridge Benchmark (Strict OOS) ---")
    
    for train_idx, test_idx in kf.split(range(N)):
        # Features for TRAIN set using only TRAIN neighbors
        G_train = G_full[np.ix_(train_idx, train_idx)] * lp_mask[np.ix_(train_idx, train_idx)]
        np.fill_diagonal(G_train, 0.0)
        X_g_train = (np.sum(G_train * y_dev_global[train_idx], axis=1) / (np.sum(G_train, axis=1) + 1e-9))
        X_train = np.hstack([X_g_train.reshape(-1, 1), X_static[train_idx]])
        
        # Features for TEST set using ONLY TRAIN neighbors
        G_test = G_full[np.ix_(test_idx, train_idx)] * lp_mask[np.ix_(test_idx, train_idx)]
        X_g_test = (np.sum(G_test * y_dev_global[train_idx], axis=1) / (np.sum(G_test, axis=1) + 1e-9))
        X_test = np.hstack([X_g_test.reshape(-1, 1), X_static[test_idx]])
        
        # Ridge Regression (L2)
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]).fit(X_train, y[train_idx])
        oos_scores.append(model.score(X_test, y[test_idx]))

    print(f"Metadata + Graph OOS R2: {np.mean(oos_scores):.4f}")

if __name__ == "__main__":
    benchmark_meta_plus_graph()
