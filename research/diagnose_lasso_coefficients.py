import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import os
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, MIGS

def diagnose_lasso():
    gt_path = "data/user_76561198039155404_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    y_dev_global = y - 5.0
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices]
    N = len(user_indices)

    # --- FEATURE ASSEMBLY (Same as before) ---
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
    
    all_graph_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)
    user_graph_vectors = all_graph_vectors[user_indices]
    dot_graph = np.dot(user_graph_vectors, user_graph_vectors.T)
    g_norms = np.linalg.norm(user_graph_vectors, axis=1)
    G_full = dot_graph / (g_norms[:, None] * g_norms[None, :] + 1e-9)
    G_full = np.maximum(0, G_full)
    np.fill_diagonal(G_full, 0.0)
    X_g = (np.sum(G_full * y_dev_global, axis=1) / (np.sum(G_full, axis=1) + 1e-9)).reshape(-1, 1)

    X = np.hstack([X_g, q_feat.reshape(-1, 1), X_meta, X_mig])
    feature_names = ["Graph_Sim", "Quality"] + meta_cols + list(MIGS.keys())

    print(f"\n--- Lasso Diagnosis (N={N}, Features={X.shape[1]}) ---")
    
    for alpha in [100.0, 10.0, 1.0, 0.1]:
        model = Lasso(alpha=alpha).fit(X, y)
        non_zero = np.sum(np.abs(model.coef_) > 1e-8)
        print(f"Alpha: {alpha:6.1f} | Non-Zero Coeffs: {non_zero}/{X.shape[1]} | Intercept: {model.intercept_:.4f}")
        if non_zero > 0:
            print("  Top Active Features:")
            indices = np.argsort(-np.abs(model.coef_))[:3]
            for idx in indices:
                if abs(model.coef_[idx]) > 1e-8:
                    print(f"    - {feature_names[idx]}: {model.coef_[idx]:.6f}")

if __name__ == "__main__":
    diagnose_lasso()
