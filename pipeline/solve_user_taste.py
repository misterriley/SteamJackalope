import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
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
    QUALITY_SCORE_S_CONST,
    GLOBAL_POSITIVE_RATE,
    W_TAG_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    TAG_NAMES_FILE
)

def solve_user_taste(ground_truth_path, output_path=None):
    """
    Solves for user preference weights using Ridge Regression.
    """
    print(f"Loading ground truth from {ground_truth_path}...")
    df_full = pd.read_csv(ground_truth_path)
    
    # Store all AppIDs to exclude them from recommendations later
    all_known_appids = set(df_full['appid'].values)
    
    # Filter for regression training: remove ignored and NaN ratings
    df = df_full.copy()
    if 'ignore' in df.columns:
        df = df[df['ignore'] == False].copy()
    
    if 'actual_rating' not in df.columns:
        print("Error: actual_rating column not found. Please verify ratings in the UI first.")
        return None
        
    df = df.dropna(subset=['actual_rating'])
    
    if len(df) < 10:
        print(f"Warning: Only {len(df)} rated games found. Results may be unstable. (Need >= 10)")
    
    user_appids = df['appid'].values
    y = df['actual_rating'].values
    
    print(f"Loading metadata and tag vectors...")
    # Get metadata for needed columns
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z', 'positive', 'negative'])
    
    # Map user appids to indices in the full dataset
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    
    if len(user_indices) != len(user_appids):
        print(f"Warning: {len(user_appids) - len(user_indices)} games in library not found in production metadata.")
        # Filter y to match found indices
        found_mask = [aid in appid_to_idx for aid in user_appids]
        y = y[found_mask]
        
    # --- DISCOVERY OPTIMIZATION ---
    # Find the optimal Discovery setting by maximizing correlation with user ratings
    print("Optimizing Discovery setting...")
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    num_steps = quality_grid.shape[0]
    
    correlations = []
    print("Step-wise Correlation Scan:")
    for i in range(num_steps):
        q_step = quality_grid[i][user_indices]
        # Calculate Pearson correlation
        if np.std(q_step) > 1e-9 and np.std(y) > 1e-9:
            corr = np.corrcoef(q_step, y)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)
        
        # Log every step
        step_disc = (i / (num_steps - 1)) * 2.0 - 1.0
        print(f"  - Step {i:2d} (Disc {step_disc:+.2f}): Absolute Correlation = {abs(corr):.4f}")
    
    best_idx = np.argmax(np.abs(correlations))
    # Map index back to -1.0 to 1.0
    optimal_disc_pref = (best_idx / (num_steps - 1)) * 2.0 - 1.0
    q_global = quality_grid[best_idx][user_indices]
    
    print(f"Optimal Discovery: {optimal_disc_pref:+.3f} (Max Absolute Correlation: {correlations[best_idx]:.4f})")

    # Load Tag Vectors (128-dim)
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    user_tag_features = tag_vectors[user_indices]
    
    # Load Metadata Features
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z']
    user_meta_features = full_metadata.iloc[user_indices][meta_cols].values
    
    # --- TAG TRANSFORMATION ---
    # Apply "Penalized Normalization" and Global Scaling (11.28)
    # This brings tag features to ~1.0 variance, matching metadata Z-scores.
    print(f"Applying penalized normalization and global scaling...")
    user_tag_norms = np.linalg.norm(user_tag_features, axis=1, keepdims=True)
    user_tag_features_norm = user_tag_features / (user_tag_norms + DOT_PRODUCT_LAMBDA)
    user_tag_features_scaled = user_tag_features_norm * TAG_GLOBAL_SCALING_FACTOR
    
    # Combine features: [Q (1)] + [Metadata (4)] + [Tags (128)]
    # Everything is now roughly Mean=0, Std=1 globally.
    X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled])
    
    # --- STABILIZATION: Add a dummy game ---
    # A game with 0 z-scores and a neutral rating (5.0) to act as a prior anchor
    dummy_X = np.zeros((1, X.shape[1]))
    dummy_y = np.array([5.0])
    X = np.vstack([X, dummy_X])
    y = np.append(y, dummy_y)
    
    print(f"Solving regression for {len(y)} samples (including dummy) across {X.shape[1]} features...")
    
    # Ridge with LOOCV - Fit directly on pre-scaled features
    # This ensures coefficients map directly to Global Z-scores.
    alphas = np.logspace(-2, 4, 50)
    model = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
    model.fit(X, y)
    
    print(f"Optimal Alpha: {model.alpha_:.4f}")
    print(f"Model R^2: {model.score(X, y):.4f}")
    
    # Extract Coefficients
    coeffs = model.coef_
    intercept = model.intercept_

    # --- TAG PROJECTION ---
    # Project whitened coefficients back to original tag space to find predictive tags
    print("Projecting coefficients back to original tag space...")
    tag_coeffs_whitened = coeffs[5:] # Skip quality + 4 metadata
    
    # Calculate Tag Norm and Unit Vector for the Recommender
    tag_norm = np.linalg.norm(tag_coeffs_whitened)
    
    if tag_norm > 1e-9:
        vibe_vector_unit = (tag_coeffs_whitened / tag_norm).tolist()
    else:
        vibe_vector_unit = tag_coeffs_whitened.tolist()

    # Load whitening matrix W (original_tags x whitened_dim)
    W = np.load(W_TAG_FILE)
    
    # Project: beta_original = W * beta_whitened
    tag_weights_original = np.dot(W, tag_coeffs_whitened)
    
    # Load Master Tag List for stable indexing
    if os.path.exists(TAG_NAMES_FILE):
        print(f"Loading master tag list from {TAG_NAMES_FILE}...")
        with open(TAG_NAMES_FILE, 'r') as f:
            unique_tags = json.load(f)
    else:
        print(f"Warning: {TAG_NAMES_FILE} not found. Falling back to metadata scan (UNSTABLE).")
        # Fallback to metadata scan
        full_metadata_tags = pd.read_parquet(METADATA_FILE, columns=['tags'])
        global_tags = set()
        for tag_str in full_metadata_tags['tags']:
            if pd.isna(tag_str) or tag_str == '' or tag_str == '[]': continue
            try:
                tags_dict = ast.literal_eval(tag_str)
                if isinstance(tags_dict, dict):
                    global_tags.update(tags_dict.keys())
            except: continue
        unique_tags = sorted(list(global_tags))
    
    if len(unique_tags) != len(tag_weights_original):
        print(f"Warning: Tag count mismatch! Names: {len(unique_tags)}, Weights: {len(tag_weights_original)}")
        # Truncate or pad to match (shouldn't happen if dataset is synced)
        min_len = min(len(unique_tags), len(tag_weights_original))
        unique_tags = unique_tags[:min_len]
        tag_weights_original = tag_weights_original[:min_len]

    # Create tag mapping and sort
    tag_impact = sorted(zip(unique_tags, tag_weights_original), key=lambda x: x[1], reverse=True)
    top_tags = [{'tag': t, 'impact': float(w)} for t, w in tag_impact[:10]]
    bottom_tags = [{'tag': t, 'impact': float(w)} for t, w in tag_impact[-10:][::-1]]

    # Load All Tag Vectors for scoring and similarity
    all_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    
    # Exclude games already in user's library or manual additions (including ignored ones)
    known_indices = [appid_to_idx[aid] for aid in all_known_appids if aid in appid_to_idx]

    # --- NORTH STAR & ABYSSAL GAMES ---
    # Find games whose TAG VECTORS (not final scores) are most similar to the taste coefficients
    print("Finding North Star and Abyssal games (pure tag alignment)...")
    vibe_vec = coeffs[5:]
    vibe_norm = np.linalg.norm(vibe_vec)
    
    if vibe_norm > 1e-9:
        # Cosine Similarity: (A . B) / (|A| * |B|)
        # Tag vectors are stored as float16, use float32 for math
        all_vectors_f32 = all_vectors.astype(np.float32)
        norms = np.linalg.norm(all_vectors_f32, axis=1)
        norms[norms == 0] = 1.0
        
        # Calculate cosine similarity to the vibe vector
        cos_sims = np.dot(all_vectors_f32, vibe_vec) / (norms * vibe_norm)
        
        # Exclude known games
        cos_sims[known_indices] = -2.0 # Way off the scale
        
        # North Stars (Highest Similarity)
        ns_indices = np.argsort(-cos_sims)[:5]
        north_stars = full_metadata.iloc[ns_indices][['appid', 'name']].copy()
        north_stars['alignment'] = cos_sims[ns_indices]
        
        # Abyssal Games (Lowest Similarity / Most Inverse)
        # We restore the known indices to mask them for the bottom search too
        cos_sims[known_indices] = 2.0
        ab_indices = np.argsort(cos_sims)[:5]
        abyssal_games = full_metadata.iloc[ab_indices][['appid', 'name']].copy()
        abyssal_games['alignment'] = cos_sims[ab_indices]
    else:
        north_stars = pd.DataFrame()
        abyssal_games = pd.DataFrame()

    # --- TOP & BOTTOM RECOMMENDATIONS ---
    print("Generating top and bottom recommendations based on solved profile...")
    # Scoring: Final Score = X_full_whitened * coeffs + intercept
    meta_features_full = full_metadata[meta_cols].values
    
    # Use the precalculated quality scores from the optimal discovery level for everyone
    q_all = quality_grid[best_idx]
    
    # Apply Penalized Norm and Global Scaling to ALL vectors
    all_norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    all_vectors_norm = all_vectors / (all_norms + DOT_PRODUCT_LAMBDA)
    all_vectors_scaled = all_vectors_norm * TAG_GLOBAL_SCALING_FACTOR
    
    X_full = np.hstack([q_all.reshape(-1, 1), full_metadata[meta_cols].values, all_vectors_scaled])
    
    # Predict directly using coefficients (Beta * Z + Intercept)
    scores = np.dot(X_full, coeffs) + intercept
    # Clamp to 0-10 scale for display
    scores = np.clip(scores, 0, 10)
    
    # Mask known games for top
    top_scores = scores.copy()
    top_scores[known_indices] = -1e12
    top_indices = np.argsort(-top_scores)[:30]
    top_games = full_metadata.iloc[top_indices][['appid', 'name']].copy()
    top_games['predicted_rating'] = top_scores[top_indices]
    
    # Mask known games for bottom
    bottom_scores = scores.copy()
    bottom_scores[known_indices] = 1e12
    bottom_indices = np.argsort(bottom_scores)[:30]
    bottom_games = full_metadata.iloc[bottom_indices][['appid', 'name']].copy()
    bottom_games['predicted_rating'] = bottom_scores[bottom_indices]
    
    # Map back to feature names
    weights = {
        'metadata': {
            'quality': float(coeffs[0]),
            'age': float(coeffs[1]),
            'popularity': float(coeffs[2]),
            'length': float(coeffs[3]),
            'difficulty': float(coeffs[4]),
            'tag_match': float(tag_norm),
            'semantic': 1.0, # Default semantic weight
            'discovery': float(optimal_disc_pref)
        },
        'vibe_vector': vibe_vector_unit,
        'intercept': float(intercept),
        'alpha': float(model.alpha_),
        'r2': float(model.score(X, y)),
        'top_tags': top_tags,
        'bottom_tags': bottom_tags,
        'north_stars': north_stars.to_dict(orient='records'),
        'abyssal_games': abyssal_games.to_dict(orient='records'),
        'top_recommendations': top_games.to_dict(orient='records'),
        'bottom_recommendations': bottom_games.to_dict(orient='records')
    }

    # Clean NaN values for JSON safety
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return obj

    weights = clean_nan(weights)
    
    # Print Summary
    print("\n--- Solved Slider Weights ---")
    for k, v in weights['metadata'].items():
        print(f"{k.capitalize():12}: {v:+.4f}")
        
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(weights, f, indent=4)
        print(f"\nUser Taste Profile saved to {output_path}")
        
    return weights

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/solve_user_taste.py <ground_truth_csv_or_steamid>")
        sys.exit(1)
        
    input_val = sys.argv[1]
    
    # Handle being passed a full path or just the ID
    if input_val.endswith('.csv'):
        gt_file = input_val
        steamid = os.path.basename(gt_file).replace('user_', '').replace('_ground_truth.csv', '')
    else:
        # Assume it's a SteamID
        steamid = input_val
        gt_file = f"data/user_{steamid}_ground_truth.csv"
        
    output = f"data/user_{steamid}_taste_profile.json"
    
    solve_user_taste(gt_file, output_path=output)
