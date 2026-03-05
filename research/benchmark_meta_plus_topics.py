import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold
import os
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR,
    TOPIC_DISTRIBUTIONS_FILE, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, MIGS

def benchmark_meta_plus_topics(user_id="76561198039155404"):
    gt_path = f"data/user_{user_id}_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices]
    N = len(user_indices)

    # --- METADATA FEATURES ---
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

    # --- TOPIC FEATURES (235-DIM ZCA-WHITENED) ---
    all_topics = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r').astype(np.float32)
    user_topics = all_topics[user_indices]
    
    # Combine X: [Metadata (42 dims) + Topics (235 dims)] = 277 total features
    X = np.hstack([X_static, user_topics])

    # --- STRICT OOS LOOP ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_scores, lasso_scores, en_scores = [], [], []
    
    print(f"\n--- Metadata + Topics Benchmark (Strict OOS) ---")
    print(f"Features: Metadata ({X_static.shape[1]}) + Raw Topics ({user_topics.shape[1]}) = {X.shape[1]} total.")
    
    for train_idx, test_idx in kf.split(range(N)):
        # Ridge (L2)
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]).fit(X[train_idx], y[train_idx])
        ridge_scores.append(ridge.score(X[test_idx], y[test_idx]))
        
        # Lasso (L1) - Use a wide range for alpha
        lasso = LassoCV(cv=5, max_iter=10000, selection='random').fit(X[train_idx], y[train_idx])
        lasso_scores.append(lasso.score(X[test_idx], y[test_idx]))
        
        # Elastic Net (Hybrid)
        en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=10000).fit(X[train_idx], y[train_idx])
        en_scores.append(en.score(X[test_idx], y[test_idx]))

    print(f"Ridge OOS R2:       {np.mean(ridge_scores):.4f}")
    print(f"Lasso OOS R2:       {np.mean(lasso_scores):.4f}")
    print(f"Elastic Net OOS R2: {np.mean(en_scores):.4f}")

if __name__ == "__main__":
    benchmark_meta_plus_topics()
