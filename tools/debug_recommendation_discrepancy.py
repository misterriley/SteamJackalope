import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.stats import norm
import os
import sys
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    QUALITY_SCORE_S_CONST,
    GLOBAL_POSITIVE_RATE,
    QUALITY_WEIGHT_MULTIPLIER,
    AGE_WEIGHT_MULTIPLIER,
    POPULARITY_WEIGHT_MULTIPLIER,
    LENGTH_WEIGHT_MULTIPLIER,
    DIFFICULTY_WEIGHT_MULTIPLIER,
    DOT_PRODUCT_LAMBDA,
    TAG_NORMS_FILE
)

def debug_discrepancy(steam_id):
    gt_path = f"data/user_{steam_id}_ground_truth.csv"
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth not found at {gt_path}")
        return

    print(f"--- DIAGNOSTIC FOR USER {steam_id} ---")
    df_gt = pd.read_csv(gt_path)
    df_gt = df_gt[df_gt['ignore'] == False].dropna(subset=['actual_rating'])
    
    print(f"Rated games in library: {len(df_gt)}")
    
    # 1. LOAD DATA
    print("Loading database...")
    full_metadata = pd.read_parquet(METADATA_FILE)
    all_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    all_norms = np.load(TAG_NORMS_FILE)
    
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    y = df_gt['actual_rating'].values
    
    # 2. SOLVER REPRODUCTION
    print("\n[STEP 1: REPRODUCTION OF SOLVER]")
    # Quality Z-scores (Bayesian)
    p = full_metadata.iloc[user_indices]['positive'].fillna(0).values
    n = full_metadata.iloc[user_indices]['negative'].fillna(0).values
    s, a = QUALITY_SCORE_S_CONST, GLOBAL_POSITIVE_RATE
    prob = (p + s * a) / (p + n + s)
    q_global = norm.ppf(np.clip(prob, 1e-6, 1 - 1e-6))
    
    # Metadata Features
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z']
    user_meta = full_metadata.iloc[user_indices][meta_cols].values
    
    # Tag Features (Penalized Norm)
    user_tags = all_vectors[user_indices].astype(np.float32)
    user_norms = all_norms[user_indices].astype(np.float32)
    user_tags_pn = user_tags / (user_norms[:, np.newaxis] + DOT_PRODUCT_LAMBDA)
    
    X = np.hstack([q_global.reshape(-1, 1), user_meta, user_tags_pn])
    
    # Stabilization Dummy
    X_train = np.vstack([X, np.zeros((1, X.shape[1]))])
    y_train = np.append(y, 5.0)
    
    # Ridge
    alphas = np.logspace(-4, 4, 50)
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)
    
    coeffs = model.coef_
    intercept = model.intercept_
    print(f"Solved R^2: {model.score(X_train, y_train):.4f}")
    print(f"Solved Weights: Q={coeffs[0]:.3f}, Date={coeffs[1]:.3f}, Pop={coeffs[2]:.3f}, Play={coeffs[3]:.3f}, Diff={coeffs[4]:.3f}")

    # Solve Top 10 using Solver Path
    print("\nGenerating Solver Top 5...")
    p_all = full_metadata['positive'].fillna(0).values
    n_all = full_metadata['negative'].fillna(0).values
    prob_all = (p_all + s * a) / (p_all + n_all + s)
    q_all = norm.ppf(np.clip(prob_all, 1e-6, 1 - 1e-6))
    
    all_tags_pn = all_vectors.astype(np.float32) / (all_norms[:, np.newaxis].astype(np.float32) + DOT_PRODUCT_LAMBDA)
    X_full = np.hstack([q_all.reshape(-1, 1), full_metadata[meta_cols].values, all_tags_pn])
    
    solver_scores = np.dot(X_full, coeffs) + intercept
    solver_scores[user_indices] = -1e12 # Exclude library
    
    top_solver_idx = np.argsort(-solver_scores)[:5]
    for i, idx in enumerate(top_solver_idx):
        print(f"{i+1}. {full_metadata.iloc[idx]['name']} (Score: {solver_scores[idx]:.3f})")

    # 3. RECOMMENDER REPRODUCTION
    print("\n[STEP 2: REPRODUCTION OF RECOMMENDER]")
    # Recommender params as set by App.tsx
    vibe_vector = coeffs[5:]
    quality_pref = coeffs[0] / QUALITY_WEIGHT_MULTIPLIER
    age_pref = coeffs[1] / AGE_WEIGHT_MULTIPLIER
    pop_pref = coeffs[2] / POPULARITY_WEIGHT_MULTIPLIER
    length_pref = coeffs[3] / LENGTH_WEIGHT_MULTIPLIER
    difficulty_pref = coeffs[4] / DIFFICULTY_WEIGHT_MULTIPLIER
    
    # Backend Scoring
    w_spps = QUALITY_WEIGHT_MULTIPLIER * quality_pref
    w_date = AGE_WEIGHT_MULTIPLIER * age_pref
    w_pop = POPULARITY_WEIGHT_MULTIPLIER * pop_pref
    w_length = LENGTH_WEIGHT_MULTIPLIER * length_pref
    w_difficulty = DIFFICULTY_WEIGHT_MULTIPLIER * difficulty_pref
    w_tag = 1.0 # Applied DNA magnitude is in the vibe_vector itself
    
    # Rec Tags: Dot(X, W) / (|X| + lambda)
    rec_tag_scores = np.dot(all_vectors.astype(np.float32), vibe_vector) / (all_norms.astype(np.float32) + DOT_PRODUCT_LAMBDA)
    
    # Rec Metadata
    rec_q = q_all 
    rec_meta = full_metadata[meta_cols].values
    
    rec_scores = (
        (rec_tag_scores * w_tag) +
        (rec_q * w_spps) +
        (rec_meta[:, 0] * w_date) +
        (rec_meta[:, 1] * w_pop) +
        (rec_meta[:, 2] * w_length) +
        (rec_meta[:, 3] * w_difficulty)
    )
    rec_scores[user_indices] = -1e12
    
    top_rec_idx = np.argsort(-rec_scores)[:5]
    print("Generating Recommender Top 5...")
    for i, idx in enumerate(top_rec_idx):
        print(f"{i+1}. {full_metadata.iloc[idx]['name']} (Score: {rec_scores[idx]:.3f})")

    # 4. ERROR ANALYSIS
    print("\n--- ERROR ANALYSIS ---")
    matches = set(top_solver_idx) == set(top_rec_idx)
    print(f"Top 5 Identical: {matches}")
    
    # Find rank of Solver #1 in Recommender list
    s1_idx = top_solver_idx[0]
    s1_rec_rank = np.where(np.argsort(-rec_scores) == s1_idx)[0][0] + 1
    print(f"Solver Top #1 ({full_metadata.iloc[s1_idx]['name']}) is Rank #{s1_rec_rank} in Recommender")
    
    # Score diff analysis
    print(f"\nScore Breakdown for Solver #1 ({full_metadata.iloc[s1_idx]['name']}):")
    print(f"  Solver Total: {solver_scores[s1_idx]:.4f}")
    print(f"  Rec Total   : {rec_scores[s1_idx]:.4f}")
    
    v = all_vectors[s1_idx]
    n = all_norms[s1_idx]
    tag_contrib = np.dot(v, vibe_vector) / (n + DOT_PRODUCT_LAMBDA)
    print(f"  Tag Contrib : {tag_contrib:.4f}")
    print(f"  Q Contrib   : {q_all[s1_idx] * w_spps:.4f}")

if __name__ == "__main__":
    debug_discrepancy("76561198039155404")
