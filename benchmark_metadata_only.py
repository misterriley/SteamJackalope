import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, MIGS

def benchmark_metadata_only(user_id="76561198039155404"):
    gt_path = f"data/user_{user_id}_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices]
    N = len(user_indices)

    # --- METADATA FEATURE ASSEMBLY ---
    # 1. Quality (Baseline Grid 0)
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_feat = to_z(quality_grid[0][user_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    
    # 2. Global Meta Columns
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z', 'tone_z']
    X_meta = np.zeros((N, len(meta_cols)))
    for j, col in enumerate(meta_cols):
        X_meta[:, j] = to_z(user_meta_df[col].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    
    # 3. Structural Metadata (MIGs - Mechanical Identity Groups)
    mig_mask_array = np.zeros((len(full_metadata), len(MIGS)), dtype=bool)
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for j, (group, tags) in enumerate(MIGS.items()):
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            mig_mask_array[:, j] |= tag_series_full.str.contains(pattern, regex=True).values
    X_mig = mig_mask_array[user_indices].astype(float)

    # Combine X: [Quality, Age, Pop, Length, Diff, Price, Tone, MIGs...]
    X = np.hstack([q_feat.reshape(-1, 1), X_meta, X_mig])

    # --- STRICT OOS LOOP ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oos_scores = []
    
    print(f"\n--- Metadata-Only OLS Benchmark (Strict OOS) ---")
    print(f"Features: Quality, Age, Pop, Length, Difficulty, Price, Tone, {X_mig.shape[1]} MIGs")
    
    for train_idx, test_idx in kf.split(range(N)):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
        # Simple Linear Regression (Ordinary Least Squares)
        model = LinearRegression().fit(X_train_scaled, y[train_idx])
        oos_scores.append(model.score(X_test_scaled, y[test_idx]))

    print(f"Metadata-Only OOS R2: {np.mean(oos_scores):.4f}")

if __name__ == "__main__":
    benchmark_metadata_only()
