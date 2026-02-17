import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
import os
import sys
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    QUALITY_GRID_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR
)

def test_r2(gt_path, dimensions):
    df = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    user_appids = df['appid'].values
    y = df['actual_rating'].values
    
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z'])
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    
    # Filter for games that actually exist in the production metadata
    mask = [aid in appid_to_idx for aid in user_appids]
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    y = y[mask]
    
    if len(y) == 0: return 0.0

    # Load Tag Vectors and Slice
    tag_vectors_full = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_vectors = tag_vectors_full[user_indices, :dimensions]
    
    # Load Metadata
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z']
    user_meta_features = full_metadata.iloc[user_indices][meta_cols].values
    
    # Quality score (Step 10 is Disc 0.0)
    quality_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
    q_global = quality_grid[10][user_indices]
    
    # Scale Tags
    user_tag_norms = np.linalg.norm(tag_vectors.astype(np.float32), axis=1, keepdims=True)
    user_tag_features_norm = tag_vectors / (user_tag_norms + DOT_PRODUCT_LAMBDA)
    user_tag_features_scaled = user_tag_features_norm * TAG_GLOBAL_SCALING_FACTOR
    
    # Combine Features
    X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled])
    
    # Ridge with LOOCV
    alphas = np.logspace(-2, 4, 50)
    model = RidgeCV(alphas=alphas)
    model.fit(X, y)
    
    return model.score(X, y)

def run_experiment():
    thresholds = {
        '80%': 170,
        '95%': 303,
        '99%': 383,
        '100%': 454
    }
    
    datasets = {
        '50 Samples': 'data/user_76561198039155404_test_50.csv',
        '150 Samples': 'data/user_76561198039155404_test_150.csv',
        'Full (382)': 'data/user_76561198039155404_ground_truth.csv'
    }
    
    results = {}
    
    for ds_name, ds_path in datasets.items():
        results[ds_name] = {}
        for th_name, dims in thresholds.items():
            r2 = test_r2(ds_path, dims)
            results[ds_name][th_name] = r2
            print(f"Dataset: {ds_name:12} | Threshold: {th_name:4} ({dims:3} dims) | R^2: {r2:.4f}")

    print("\n--- FINAL R^2 GRID ---")
    header = "Dataset      | 80% Var | 95% Var | 99% Var | 100% Var"
    print(header)
    print("-" * len(header))
    for ds_name in datasets:
        row = f"{ds_name:12} | "
        row_vals = [f"{results[ds_name][th]:.4f}" for th in ['80%', '95%', '99%', '100%']]
        row += " | ".join(row_vals)
        print(row)

if __name__ == "__main__":
    run_experiment()
