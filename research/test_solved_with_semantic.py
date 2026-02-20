import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import os
import sys
import json
import ast
import re

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    QUALITY_SCORE_S_CONST,
    GLOBAL_POSITIVE_RATE,
    W_TAG_FILE,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    TAG_NAMES_FILE,
    TAG_PRIOR_COUNTS_FILE,
    TAG_PRIOR_TRANSFORMED_FILE,
    ADAPTIVE_DNA_BASE_K,
    ADAPTIVE_DNA_SLOPE,
    DIFFICULTY_NEUTRAL_FALLBACK,
    DNA_UI_SCALING_FACTOR,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR
)

def solve_user_taste_experimental(ground_truth_path, output_path=None, use_semantic=True):
    """
    Experimental solver using both Tag and (optionally) Semantic dimensions.
    Matches the logic of pipeline/solve_user_taste.py exactly but adds semantic features to X.
    """
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    
    sl_path = ground_truth_path.replace('_ground_truth.csv', '_soft_labels.csv')
    if os.path.exists(sl_path):
        df_sl = pd.read_csv(sl_path)
        all_library_appids = df_sl['appid'].unique().tolist()
    else:
        all_library_appids = df_gt['appid'].unique().tolist()
    
    ignored_appids = []
    if 'ignore' in df_gt.columns:
        ignored_appids = df_gt[df_gt['ignore'] == True]['appid'].tolist()

    df = df_gt.copy()
    if 'ignore' in df.columns:
        df = df[df['ignore'] == False].copy()
    
    df = df.dropna(subset=['actual_rating'])
    user_appids = df['appid'].values
    y = df['actual_rating'].values
    
    print(f"Loading metadata and features...")
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z', 'price_z', 'positive', 'negative', 'tags', 'release_year', 'difficulty_predicted', 'price'])
    
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    
    if len(user_indices) != len(user_appids):
        found_mask = [aid in appid_to_idx for aid in user_appids]
        y = y[found_mask]
        
    # --- DISCOVERY OPTIMIZATION ---
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    num_steps = quality_grid.shape[0]
    correlations = []
    for i in range(num_steps):
        q_step = np.clip(quality_grid[i][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
        if np.std(q_step) > 1e-9 and np.std(y) > 1e-9:
            corr = np.corrcoef(q_step, y)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)
    
    best_idx = np.argmax(np.abs(correlations))
    optimal_disc_pref = (best_idx / (num_steps - 1)) * 2.0 - 1.0
    q_global = np.clip(quality_grid[best_idx][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # --- TAG FEATURES ---
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    full_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    
    num_ratings = len(y)
    k_max = tag_vectors.shape[1]
    k_adaptive = int(np.clip(num_ratings - 7, 1, k_max))
    
    user_tag_norms = full_norms[user_indices].reshape(-1, 1).astype(np.float32)
    user_tag_features = tag_vectors[user_indices, :k_adaptive].astype(np.float32)
    user_tag_features_scaled = (user_tag_features / (user_tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    # --- METADATA FEATURES ---
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
    user_meta_features = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # --- SEMANTIC FEATURES (OPTIONAL) ---
    X_parts = [q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled]
    
    if use_semantic:
        print("Loading semantic features...")
        sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
        sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
        
        user_sem_norms = sem_norms[user_indices].reshape(-1, 1).astype(np.float32)
        user_sem_features = sem_vectors[user_indices].astype(np.float32)
        # Use a boosted scaling factor for this test to match tag variance
        EXPERIMENTAL_SEM_SCALING = 11.25
        user_sem_features_scaled = (user_sem_features / (user_sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * EXPERIMENTAL_SEM_SCALING
        X_parts.append(user_sem_features_scaled)
        
    X = np.hstack(X_parts)
    
    # --- STABILIZATION ---
    dummy_X = np.zeros((1, X.shape[1]))
    dummy_y = np.array([DIFFICULTY_NEUTRAL_FALLBACK])
    X_train = np.vstack([X, dummy_X])
    y_train = np.append(y, dummy_y)
    
    print(f"Solving Lasso regression for {len(y_train)} samples across {X.shape[1]} features...")
    model = LassoCV(cv=5, max_iter=20000, selection='random', tol=1e-3)
    model.fit(X_train, y_train)
    
    coeffs = model.coef_
    intercept = model.intercept_
    
    print(f"Training R^2: {model.score(X_train, y_train):.4f}")
    
    # --- RESULT PREPARATION (SIMPLIFIED FOR RESEARCH) ---
    q_weight = float(coeffs[0])
    meta_weights = coeffs[1:6].tolist()
    tag_coeffs_adaptive = coeffs[6:6+k_adaptive]
    
    tag_norm = np.linalg.norm(tag_coeffs_adaptive)
    
    sem_weight_total = 0.0
    if use_semantic:
        sem_coeffs = coeffs[6+k_adaptive:]
        sem_weight_total = np.linalg.norm(sem_coeffs)
        print(f"Semantic weight norm: {sem_weight_total:.4f}")

    # --- TOP PREDICTIONS ---
    print("Generating top predictions...")
    from common.utils import calculate_linear_scores
    
    # Padded tag vector for calculate_linear_scores
    tag_coeffs_full = np.zeros(k_max)
    tag_coeffs_full[:k_adaptive] = tag_coeffs_adaptive
    
    preview_weights = {
        'quality': q_weight,
        'age': meta_weights[0],
        'popularity': meta_weights[1],
        'length': meta_weights[2],
        'difficulty': meta_weights[3],
        'price': meta_weights[4],
        'tag_match': float(tag_norm),
        'semantic': float(sem_weight_total) if use_semantic else 1.0,
        'discovery': float(optimal_disc_pref)
    }
    
    # Base scores (Quality + Meta + Tags)
    scores = calculate_linear_scores(
        z_quality=quality_grid[best_idx],
        z_date=full_metadata['date_z'].values,
        z_pop=full_metadata['pop_z'].values,
        z_playtime=full_metadata['playtime_z'].values,
        z_difficulty=full_metadata['difficulty_z'].values,
        z_price=full_metadata['price_z'].values,
        tag_vectors=tag_vectors,
        tag_norms=full_norms,
        beta_tag=tag_coeffs_full,
        weights=preview_weights,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR,
        dot_product_lambda=DOT_PRODUCT_LAMBDA,
        z_clamp_min=Z_SCORE_CLAMP_MIN,
        z_clamp_max=Z_SCORE_CLAMP_MAX,
        dna_scaling_factor=1.0,
        intercept=float(intercept)
    )
    
    # Add Semantic if used
    if use_semantic:
        sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
        sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').reshape(-1, 1).astype(np.float32)
        
        EXPERIMENTAL_SEM_SCALING = 11.25
        batch_size = 50000
        sem_scores = np.zeros(len(full_metadata))
        for i in range(0, len(full_metadata), batch_size):
            end = min(i + batch_size, len(full_metadata))
            batch_vecs = sem_vectors[i:end].astype(np.float32)
            batch_scaled = (batch_vecs / (sem_norms[i:end] + SEMANTIC_DOT_PRODUCT_LAMBDA)) * EXPERIMENTAL_SEM_SCALING
            sem_scores[i:end] = np.dot(batch_scaled, coeffs[6+k_adaptive:])
        
        scores += sem_scores

    # Mask known games
    known_indices = [appid_to_idx[aid] for aid in all_library_appids if aid in appid_to_idx]
    scores[known_indices] = -1e12
    
    # Get top 20
    top_indices = np.argsort(-scores)[:20]
    top_results = full_metadata.iloc[top_indices][['appid', 'name']].copy()
    top_results['predicted_rating'] = scores[top_indices]
    
    print("\n--- Weights ---")
    print(f"Quality:    {q_weight:+.4f}")
    print(f"Age (Date): {meta_weights[0]:+.4f}")
    print(f"Popularity: {meta_weights[1]:+.4f}")
    print(f"Length:     {meta_weights[2]:+.4f}")
    print(f"Difficulty: {meta_weights[3]:+.4f}")
    print(f"Price:      {meta_weights[4]:+.4f}")
    print(f"Tag Norm:   {tag_norm:+.4f}")
    if use_semantic:
        print(f"Sem Norm:   {sem_weight_total:+.4f}")

    print("\nTop 20 Predicted Games (Excluding Library):")
    print(top_results.to_string(index=False))

if __name__ == "__main__":
    user_id = "76561198039155404"
    gt_file = f"data/user_{user_id}_ground_truth.csv"
    
    print("--- RUNNING WITH SEMANTIC DATA ---")
    solve_user_taste_experimental(gt_file, use_semantic=True)
