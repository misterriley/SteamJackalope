import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
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
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    SEMANTIC_GLOBAL_SCALING_FACTOR,
    DIFFICULTY_NEUTRAL_FALLBACK,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)

def test_semantic_regression(user_id):
    ground_truth_path = f"data/user_{user_id}_ground_truth.csv"
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    
    df = df_gt.copy()
    if 'ignore' in df.columns:
        df = df[df['ignore'] == False].copy()
    
    df = df.dropna(subset=['actual_rating'])
    num_ratings = len(df)
    user_appids = df['appid'].values
    y = df['actual_rating'].values
    
    print(f"Rated games: {num_ratings}")
    
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z', 'price_z'])
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    q_global = np.clip(quality_grid[10][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
    user_meta_features = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    full_tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    user_tag_features_raw = tag_vectors[user_indices].astype(np.float32)
    user_tag_norms = full_tag_norms[user_indices].reshape(-1, 1).astype(np.float32)
    user_tag_features_scaled = (user_tag_features_raw / (user_tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    semantic_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    semantic_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    user_sem_features_raw = semantic_vectors[user_indices].astype(np.float32)
    user_sem_norms = semantic_norms[user_indices].reshape(-1, 1).astype(np.float32)
    user_sem_features_scaled = (user_sem_features_raw / (user_sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
    
    X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled, user_sem_features_scaled])
    
    # Scale X locally for the experiment
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add dummy
    dummy_X = np.zeros((1, X_scaled.shape[1]))
    dummy_y = np.array([DIFFICULTY_NEUTRAL_FALLBACK])
    X_train = np.vstack([X_scaled, dummy_X])
    y_train = np.append(y, dummy_y)
    
    print(f"Running LASSO across {X_train.shape[1]} features (Standardized)...")
    model = LassoCV(cv=5, max_iter=20000, selection='random', tol=1e-3)
    model.fit(X_train, y_train)
    
    coeffs = model.coef_
    intercept = model.intercept_
    
    q_weight = coeffs[0]
    meta_weights = coeffs[1:6]
    tag_weights = coeffs[6:6+231]
    sem_weights = coeffs[6+231:]
    
    print(f"\nTraining R^2: {model.score(X_train, y_train):.4f}")
    
    print("\n--- Weights (Standardized Space) ---")
    print(f"Quality:    {q_weight:+.4f}")
    for i, col in enumerate(meta_cols):
        print(f"{col:11}: {meta_weights[i]:+.4f}")
    
    print(f"Tags:       {np.sum(np.abs(tag_weights) > 1e-6):4d} active features. Sum Abs Weight: {np.sum(np.abs(tag_weights)):.4f}")
    print(f"Semantic:   {np.sum(np.abs(sem_weights) > 1e-6):4d} active features. Sum Abs Weight: {np.sum(np.abs(sem_weights)):.4f}")
    
    # --- PREDICTIONS ---
    print("\nCalculating predictions for ALL games...")
    # We need to apply the scaler to the full population
    # To do this memory-efficiently, we'll do it in chunks
    
    batch_size = 50000
    all_final_scores = np.zeros(len(full_metadata))
    all_tag_scores = np.zeros(len(full_metadata))
    all_sem_scores = np.zeros(len(full_metadata))
    
    for i in range(0, len(full_metadata), batch_size):
        end = min(i + batch_size, len(full_metadata))
        
        batch_q = np.clip(quality_grid[10][i:end], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX).reshape(-1, 1)
        batch_meta = np.clip(full_metadata.iloc[i:end][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
        
        batch_tag_raw = tag_vectors[i:end].astype(np.float32)
        batch_tag_norms = full_tag_norms[i:end].reshape(-1, 1).astype(np.float32)
        batch_tag_scaled = (batch_tag_raw / (batch_tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
        
        batch_sem_raw = semantic_vectors[i:end].astype(np.float32)
        batch_sem_norms = semantic_norms[i:end].reshape(-1, 1).astype(np.float32)
        batch_sem_scaled = (batch_sem_raw / (batch_sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
        
        batch_X = np.hstack([batch_q, batch_meta, batch_tag_scaled, batch_sem_scaled])
        batch_X_std = scaler.transform(batch_X)
        
        batch_scores = np.dot(batch_X_std, coeffs) + intercept
        all_final_scores[i:end] = batch_scores
        
        # Contribution analysis (using scaled coeffs)
        # Note: standardized coeff beta_std relates to raw coeff beta_raw via beta_std = beta_raw * sigma_x
        # So contribution is (x - mu)/sigma * beta_std
        all_tag_scores[i:end] = np.dot(batch_X_std[:, 6:6+231], tag_weights)
        all_sem_scores[i:end] = np.dot(batch_X_std[:, 6+231:], sem_weights)
        
    # Mask library
    known_indices = [appid_to_idx[aid] for aid in df_gt['appid'].values if aid in appid_to_idx]
    all_final_scores[known_indices] = -1e12
    
    top_indices = np.argsort(-all_final_scores)[:20]
    results = full_metadata.iloc[top_indices][['appid', 'name']].copy()
    results['predicted_rating'] = all_final_scores[top_indices]
    results['tag_contrib'] = all_tag_scores[top_indices]
    results['sem_contrib'] = all_sem_scores[top_indices]
    
    print("\nTop 20 Predicted Games (Standardized Experiment):")
    print(results.to_string(index=False))

if __name__ == "__main__":
    test_semantic_regression("76561198039155404")
