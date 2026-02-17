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
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    TAG_NAMES_FILE,
    TAG_PRIOR_COUNTS_FILE,
    TAG_PRIOR_TRANSFORMED_FILE,
    ADAPTIVE_DNA_BASE_K,
    ADAPTIVE_DNA_SLOPE
)

def solve_user_taste(ground_truth_path, output_path=None):
    """
    Solves for user preference weights using Ridge Regression.
    """
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    
    # Load full library from soft labels to ensure ALL owned games (including zero playtime)
    # are tracked for exclusion, even if they aren't in the training set.
    sl_path = ground_truth_path.replace('_ground_truth.csv', '_soft_labels.csv')
    if os.path.exists(sl_path):
        print(f"Loading full library list from {sl_path}...")
        df_sl = pd.read_csv(sl_path)
        all_library_appids = df_sl['appid'].unique().tolist()
    else:
        all_library_appids = df_gt['appid'].unique().tolist()
    
    # Filter for regression training: remove ignored and NaN ratings
    df = df_gt.copy()
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
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z', 'positive', 'negative', 'tags'])
    
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
    from common.constants import Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
    
    for i in range(num_steps):
        # Apply Clamping to match Recommender exactly during optimization
        q_step = np.clip(quality_grid[i][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
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
    # Apply Clamping to the best grid row
    q_global = np.clip(quality_grid[best_idx][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    print(f"Optimal Discovery: {optimal_disc_pref:+.3f} (Max Absolute Correlation: {correlations[best_idx]:.4f})")

    # Load Tag Vectors (Current production K, e.g., 243-dim)
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    user_tag_features_raw = tag_vectors[user_indices]
    
    # --- ADAPTIVE DIMENSIONALITY ---
    # Based on parametric study, K = N - 6 reaches the saturation point for df
    num_ratings = len(y) - 1 # Exclude dummy
    
    k_max = tag_vectors.shape[1] # Production max (e.g. 243)
    k_adaptive = int(np.clip(num_ratings - 6, 1, k_max))
    
    print(f"Adaptive DNA: Using saturation dimensionality K = {k_adaptive} for library size {num_ratings}.")
    
    # 1. Use the FULL norm for penalized normalization (consistency with Recommender)
    full_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    user_tag_norms = full_norms[user_indices].reshape(-1, 1).astype(np.float32)
    
    # 2. Slice features to the adaptive dimensionality
    user_tag_features = user_tag_features_raw[:, :k_adaptive].astype(np.float32)
    
    # Load Metadata Features
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z']
    # Apply Clamping to match Recommender exactly during training
    user_meta_features = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # --- TAG TRANSFORMATION ---
    print(f"Applying penalized normalization and global scaling...")
    user_tag_features_norm = user_tag_features / (user_tag_norms + DOT_PRODUCT_LAMBDA)
    user_tag_features_scaled = user_tag_features_norm * TAG_GLOBAL_SCALING_FACTOR
    
    # Combine features: [Q (1)] + [Metadata (4)] + [Tags (k)]
    X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled])
    
    # --- STABILIZATION: Add a dummy game ---
    dummy_X = np.zeros((1, X.shape[1]))
    dummy_y = np.array([5.0])
    X = np.vstack([X, dummy_X])
    y = np.append(y, dummy_y)
    
    print(f"Solving Lasso regression for {len(y)} samples across {X.shape[1]} features...")
    
    from sklearn.linear_model import LassoCV
    # Use LassoCV with a broad alpha range and high iterations for stability
    # Note: We NO LONGER use StandardScaler here, as the features are already 
    # standardized to Global Z-scores (Metadata) or Scaled Norms (Tags).
    model = LassoCV(cv=5, max_iter=20000, selection='random', tol=1e-3)
    model.fit(X, y)
    
    # Extract Coefficients directly
    coeffs = model.coef_
    intercept = model.intercept_
    
    print(f"Optimal Alpha: {model.alpha_:.4f}")
    # Calculate R^2 on the training set
    r2_train = model.score(X, y)
    print(f"Model Training R^2: {r2_train:.4f}")
    
    # Extract Coefficients
    coeffs = model.coef_
    intercept = model.intercept_

    # --- TAG PROJECTION ---
    # Project whitened coefficients back to original tag space to find predictive tags
    print("Projecting coefficients back to original tag space...")
    tag_coeffs_adaptive = coeffs[5:] # Skip quality + 4 metadata
    
    # PAD coefficients back to the full production K (e.g., 243)
    full_k = tag_vectors.shape[1]
    tag_coeffs_full = np.zeros(full_k)
    tag_coeffs_full[:k_adaptive] = tag_coeffs_adaptive
    
    # Create full coefficient vector for scoring: [Q] + [4 Metadata] + [Full Tags]
    full_coeffs = np.zeros(5 + full_k)
    full_coeffs[:5] = coeffs[:5] # Quality + Metadata
    full_coeffs[5:] = tag_coeffs_full
    
    # Calculate Tag Norm and Unit Vector for the Recommender (using padded full vector)
    tag_norm = np.linalg.norm(tag_coeffs_full)
    
    if tag_norm > 1e-9:
        vibe_vector_unit = (tag_coeffs_full / tag_norm).tolist()
    else:
        vibe_vector_unit = tag_coeffs_full.tolist()

    # Load whitening matrix W (original_tags x whitened_dim)
    W = np.load(W_TAG_FILE)
    
    # Project: beta_original = W * beta_whitened
    # Using the full padded vector ensures we multiply by the correct columns of W
    tag_weights_original = np.dot(W, tag_coeffs_full)
    
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
                # Basic cleanup to avoid ast.literal_eval slowness if possible
                if '{' in tag_str:
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

    # --- SANITY CHECK: Support-Based Filtering ---
    # We calculate how many games in the user's library actually contain each tag.
    # If a tag has zero support, it's likely a statistical alias and should be masked from the UI.
    print("Calculating tag support in user library for sanity check...")
    user_tags_raw = full_metadata.iloc[user_indices]['tags'].values
    tag_support = np.zeros(len(unique_tags), dtype=int)
    tag_to_idx_map = {tag: i for i, tag in enumerate(unique_tags)}
    
    for tag_str in user_tags_raw:
        if pd.isna(tag_str) or tag_str == '' or tag_str == '[]': continue
        try:
            # Using ast.literal_eval for safety
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                for tag in tags_dict.keys():
                    if tag in tag_to_idx_map:
                        tag_support[tag_to_idx_map[tag]] += 1
        except: continue
        
    # Threshold: Must appear in at least N games. 
    # For now, N=1 is sufficient to kill "bogus" associations.
    SUPPORT_THRESHOLD = 1
    support_mask = tag_support >= SUPPORT_THRESHOLD
    
    # Filter the impacts: only tags with sufficient support are eligible for top/bottom lists
    eligible_impacts = [
        (t, float(w)) for i, (t, w) in enumerate(zip(unique_tags, tag_weights_original)) 
        if support_mask[i]
    ]
    
    # Sort the supported impacts
    tag_impact = sorted(eligible_impacts, key=lambda x: x[1], reverse=True)
    top_tags = [{'tag': t, 'impact': w} for t, w in tag_impact[:10]]
    bottom_tags = [{'tag': t, 'impact': w} for t, w in tag_impact[-10:][::-1]]

    # Load All Tag Vectors for scoring and similarity
    all_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    
    # Exclude games already in user's library or manual additions (including ignored ones)
    known_indices = [appid_to_idx[aid] for aid in all_library_appids if aid in appid_to_idx]

    # --- NORTH STAR & ABYSSAL GAMES ---
    # Find games whose TAG VECTORS (not final scores) are most similar to the taste coefficients
    print("Finding North Star and Abyssal games (pure tag alignment)...")
    vibe_vec = tag_coeffs_full
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
    print("Generating top and bottom recommendations based on solved profile (unified pathway)...")
    # Using unified scoring utility for bit-perfect parity with Recommender
    from common.utils import calculate_linear_scores
    
    # Restore absolute tag coefficients
    beta_tag_absolute = tag_coeffs_full
    
    # Define weights dictionary for utility
    meta_weights = {
        'quality': float(coeffs[0]),
        'age': float(coeffs[1]),
        'popularity': float(coeffs[2]),
        'length': float(coeffs[3]),
        'difficulty': float(coeffs[4])
    }
    
    # Predict directly using unified function
    # Note: Quality grid is already loaded as quality_grid
    all_tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    
    scores = calculate_linear_scores(
        z_quality=quality_grid[best_idx],
        z_date=full_metadata['date_z'].values,
        z_pop=full_metadata['pop_z'].values,
        z_playtime=full_metadata['playtime_z'].values,
        z_difficulty=full_metadata['difficulty_z'].values,
        tag_vectors=all_vectors,
        tag_norms=all_tag_norms,
        beta_tag=beta_tag_absolute,
        weights=meta_weights,
        intercept=intercept,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR,
        dot_product_lambda=DOT_PRODUCT_LAMBDA,
        z_clamp_min=Z_SCORE_CLAMP_MIN,
        z_clamp_max=Z_SCORE_CLAMP_MAX
    )
    
    # Clamp to 0-10 scale for display
    scores = np.clip(scores, 0, 10)
    
    # --- APPLY DEFAULT FILTERS (Match Recommender) ---
    mask = np.ones(len(full_metadata), dtype=bool)
    # 1. English Only
    if 'is_english' in full_metadata.columns:
        mask &= full_metadata['is_english'].values
    # 2. No VR Only
    if 'is_vr_only' in full_metadata.columns:
        mask &= ~full_metadata['is_vr_only'].values
    # 3. No Utilities
    if 'is_utility' in full_metadata.columns:
        mask &= ~full_metadata['is_utility'].values
    # 4. Released Only
    if 'parsed_date' in full_metadata.columns:
        build_time = pd.Timestamp.now() # Close enough to mtime
        future_mask = (full_metadata['parsed_date'] > build_time).fillna(False)
        mask &= ~future_mask.values

    # Mask known games for top
    top_scores = scores.copy()
    top_scores[known_indices] = -1e12
    # Apply filters
    top_scores[~mask] = -1e12
    
    # Get top 30 using stable lexicographical sort (score DESC, name ASC)
    all_names = full_metadata['name'].fillna("").values
    top_indices = np.lexsort((all_names, -top_scores))[:30]
    
    top_games = full_metadata.iloc[top_indices][['appid', 'name']].copy()
    top_games['predicted_rating'] = top_scores[top_indices]
    
    # Mask known games for bottom
    bottom_scores = scores.copy()
    bottom_scores[known_indices] = 1e12
    # Apply filters (we want the "worst" of the VALID games)
    bottom_scores[~mask] = 1e12
    
    # Get bottom 30 using stable lexicographical sort (score ASC, name ASC)
    bottom_indices = np.lexsort((all_names, bottom_scores))[:30]
    
    bottom_games = full_metadata.iloc[bottom_indices][['appid', 'name']].copy()
    bottom_games['predicted_rating'] = bottom_scores[bottom_indices]
    
    # --- WEIGHT SCALING (Slider Real Estate) ---
    # We scale metadata weights so the largest absolute value is 3.0.
    # This preserves ranking order but makes better use of the UI slider range.
    weights_to_scale = {
        'quality': float(coeffs[0]),
        'age': float(coeffs[1]),
        'popularity': float(coeffs[2]),
        'length': float(coeffs[3]),
        'difficulty': float(coeffs[4]),
        'tag_match': float(tag_norm),
        'semantic': 1.0  # Default base semantic weight
    }
    
    max_abs_weight = max(abs(v) for v in weights_to_scale.values())
    scaling_factor = 3.0 / max_abs_weight if max_abs_weight > 1e-6 else 1.0
    
    scaled_metadata = {k: v * scaling_factor for k, v in weights_to_scale.items()}
    # Keep discovery separate (do not scale)
    scaled_metadata['discovery'] = float(optimal_disc_pref)
    
    # Scale intercept to maintain relative signal-to-bias ratio for the preview scores
    scaled_intercept = float(intercept) * scaling_factor

    # Map back to feature names
    weights = {
        'metadata': scaled_metadata,
        'vibe_vector': vibe_vector_unit,
        'intercept': scaled_intercept,
        'alpha': float(model.alpha_),
        'r2': float(model.score(X, y)),
        'library_appids': all_library_appids,
        'rated_appids': df_gt[~df_gt['ignore'].fillna(False) & df_gt['actual_rating'].notna()]['appid'].tolist(),
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
