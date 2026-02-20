import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import sys
import json
import ast

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX,
    DIFFICULTY_NEUTRAL_FALLBACK
)

def prepare_data(df_gt, full_metadata, tag_vectors, full_norms, quality_grid):
    user_appids = df_gt['appid'].values
    y = df_gt['actual_rating'].values
    
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    
    if len(user_indices) != len(user_appids):
        found_mask = [aid in appid_to_idx for aid in user_appids]
        y = y[found_mask]
        user_appids = user_appids[found_mask]

    # Find optimal discovery (for simplicity in CV, we use the global best or just pick step 10)
    # To be rigorous, we should optimize this on each fold, but let's stick to step 10 for now.
    q_global = np.clip(quality_grid[10][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # Adaptive DNA (K = N - 7)
    num_ratings = len(y)
    k_adaptive = int(np.clip(num_ratings - 7, 1, tag_vectors.shape[1]))
    
    user_tag_norms = full_norms[user_indices].reshape(-1, 1).astype(np.float32)
    user_tag_features = tag_vectors[user_indices, :k_adaptive].astype(np.float32)
    
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
    user_meta_features = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # Neutral Imputation for Price (Ori Fix)
    # Note: df_gt might not have price_z, we get it from full_metadata
    # Already done by indexing full_metadata above.
    
    user_tag_features_norm = user_tag_features / (user_tag_norms + DOT_PRODUCT_LAMBDA)
    user_tag_features_scaled = user_tag_features_norm * TAG_GLOBAL_SCALING_FACTOR
    
    X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled])
    
    return X, y

def run_nested_cv(X_full, y_full, use_scaler=True):
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_errors = []
    
    for train_idx, test_idx in kf_outer.split(X_full):
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        
        # Add dummy game to training set
        dummy_X = np.zeros((1, X_train.shape[1]))
        dummy_y = np.array([DIFFICULTY_NEUTRAL_FALLBACK])
        X_train_augmented = np.vstack([X_train, dummy_X])
        y_train_augmented = np.append(y_train, dummy_y)
        
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_augmented)
            X_test_scaled = scaler.transform(X_test) # Do NOT scale test based on dummy
        else:
            X_train_scaled = X_train_augmented
            X_test_scaled = X_test
            
        model = LassoCV(cv=5, max_iter=20000, selection='random', tol=1e-3)
        model.fit(X_train_scaled, y_train_augmented)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        outer_errors.append(mse)
        
    return np.mean(outer_errors), np.std(outer_errors)

if __name__ == "__main__":
    steamid = "76561198039155404"
    gt_path = f"data/user_{steamid}_ground_truth.csv"
    
    print(f"Loading data for user {steamid}...")
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    df_gt = df_gt[df_gt['ignore'] == False]
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    full_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    
    X, y = prepare_data(df_gt, full_metadata, tag_vectors, full_norms, quality_grid)
    
    print(f"Running Nested CV with StandardScaler...")
    mean_mse_scaled, std_mse_scaled = run_nested_cv(X, y, use_scaler=True)
    
    print(f"Running Nested CV WITHOUT StandardScaler...")
    mean_mse_raw, std_mse_raw = run_nested_cv(X, y, use_scaler=False)
    
    print("\n--- Results (Mean Squared Error) ---")
    print(f"With Scaler:    {mean_mse_scaled:.4f} (+/- {std_mse_scaled:.4f})")
    print(f"Without Scaler: {mean_mse_raw:.4f} (+/- {std_mse_raw:.4f})")
    
    if mean_mse_raw < mean_mse_scaled:
        print("\nHypothesis Confirmed: Removing StandardScaler improves generalization (lower MSE).")
    else:
        print("\nHypothesis Refuted: StandardScaler performs better or comparable.")
