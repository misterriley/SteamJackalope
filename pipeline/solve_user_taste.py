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
    ADAPTIVE_DNA_SLOPE,
    DIFFICULTY_NEUTRAL_FALLBACK,
    DNA_UI_SCALING_FACTOR,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
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
    all_library_appids = set()
    library_details = {} # appid -> {playtime, personalized_q, p_plus_t}
    
    if os.path.exists(sl_path):
        print(f"Loading full library list from {sl_path}...")
        df_sl = pd.read_csv(sl_path)
        all_library_appids.update(df_sl['appid'].unique().tolist())
        # Store details for personalization
        for _, row in df_sl.iterrows():
            aid = int(row['appid'])
            library_details[aid] = {
                'playtime': float(row['playtime_forever']),
                'personalized_q': float(row['personalized_q']),
                'p_plus_t': float(row['p_plus_t'])
            }
    
    # Also include anything in ground truth (manual additions, rated games)
    all_library_appids.update(df_gt['appid'].unique().tolist())
    all_library_appids = list(all_library_appids)
    
    # Filter for regression training: remove ignored and NaN ratings
    # Get list of ignored appids for the JSON
    ignored_appids = []
    if 'ignore' in df_gt.columns:
        ignored_appids = df_gt[df_gt['ignore'] == True]['appid'].tolist()

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
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z', 'price_z', 'positive', 'negative', 'tags', 'is_delisted', 'is_english', 'is_vr_only', 'is_utility', 'is_hollow', 'parsed_date'])
    
    # Ensure boolean columns are actually boolean (Parquet stores them as int8 for space)
    bool_cols = ['is_delisted', 'is_english', 'is_vr_only', 'is_utility', 'is_hollow']
    for col in bool_cols:
        if col in full_metadata.columns:
            full_metadata[col] = full_metadata[col].astype(bool)
    
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
    
    # Store correlation scan for explainability
    discovery_scan = []
    for i, corr in enumerate(correlations):
        disc_val = (i / (num_steps - 1)) * 2.0 - 1.0
        discovery_scan.append({'x': disc_val, 'y': corr**2}) # Store R^2
    
    print(f"Optimal Discovery: {optimal_disc_pref:+.3f} (Max R^2: {correlations[best_idx]**2:.4f})")

    # Load Tag Vectors (Current production K, e.g., 243-dim)
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    user_tag_features_raw = tag_vectors[user_indices]
    
    # --- ADAPTIVE DIMENSIONALITY ---
    # Based on parametric study, K = N - 7 reaches the saturation point for df
    num_ratings = len(y) - 1 # Exclude dummy
    
    k_max = tag_vectors.shape[1] # Production max (e.g. 243)
    k_adaptive = int(np.clip(num_ratings - 7, 1, k_max))
    
    print(f"Adaptive DNA: Using saturation dimensionality K = {k_adaptive} for library size {num_ratings}.")
    
    # 1. Use the FULL norm for penalized normalization (consistency with Recommender)
    full_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    user_tag_norms = full_norms[user_indices].reshape(-1, 1).astype(np.float32)
    
    # 2. Slice features to the adaptive dimensionality
    user_tag_features = user_tag_features_raw[:, :k_adaptive].astype(np.float32)
    
    # Load Metadata Features (Including price_z)
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
    # Apply Clamping to match Recommender exactly during training
    user_meta_features = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # --- TAG TRANSFORMATION ---
    print(f"Applying penalized normalization and global scaling to tags...")
    user_tag_features_norm = user_tag_features / (user_tag_norms + DOT_PRODUCT_LAMBDA)
    user_tag_features_scaled = user_tag_features_norm * TAG_GLOBAL_SCALING_FACTOR
    
    # --- SEMANTIC FEATURES ---
    print(f"Loading semantic features and applying penalized normalization...")
    semantic_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    semantic_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    
    user_sem_features_raw = semantic_vectors[user_indices].astype(np.float32)
    user_sem_norms = semantic_norms[user_indices].reshape(-1, 1).astype(np.float32)
    user_sem_features_scaled = (user_sem_features_raw / (user_sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
    
    # Combine features: [Q (1)] + [Metadata (5)] + [Tags (k)] + [Semantic (235)]
    X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled, user_sem_features_scaled])
    
    # --- STABILIZATION: Add a dummy game ---
    dummy_X = np.zeros((1, X.shape[1]))
    dummy_y = np.array([DIFFICULTY_NEUTRAL_FALLBACK])
    X = np.vstack([X, dummy_X])
    y = np.append(y, dummy_y)
    
    print(f"Solving Lasso regression for {len(y)} samples across {X.shape[1]} features...")
    
    from sklearn.linear_model import LassoCV
    
    # --- MODEL SELECTION: RAW VS SCALED ---
    # We choose NOT to use StandardScaler because the input features are already 
    # globally standardized (pop_z, date_z, whitened tags). 
    # Local scaling in a biased user sample can over-represent low-variance features 
    # that aren't actually predictive in the global population.
    
    # Use LassoCV to find the best sparse model on the RAW (globally scaled) features
    model = LassoCV(cv=5, max_iter=20000, selection='random', tol=1e-3)
    model.fit(X, y)
    
    # Coefficients are already on the correct scale
    coeffs = model.coef_
    intercept = model.intercept_
    
    print(f"Optimal Alpha: {model.alpha_:.4f}")
    # Calculate R^2 on the training set
    r2_train = model.score(X, y)
    print(f"Model Training R^2: {r2_train:.4f}")

    # --- TAG PROJECTION ---
    # Project whitened coefficients back to original tag space to find predictive tags
    print("Projecting coefficients back to original tag space...")
    tag_coeffs_adaptive = coeffs[6:6+k_adaptive]
    sem_coeffs_full = coeffs[6+k_adaptive:]
    
    # PAD coefficients back to the full production K (e.g., 243)
    full_k = tag_vectors.shape[1]
    tag_coeffs_full = np.zeros(full_k)
    tag_coeffs_full[:k_adaptive] = tag_coeffs_adaptive
    
    # Create full coefficient vector for scoring: [Q] + [5 Metadata] + [Full Tags]
    # (Optional: we can keep sem_coeffs separate for better structured response)
    
    # --- TAG DIMENSIONS ANALYSIS ---
    # Find the top 5 predictive dimensions from the whitened coefficients
    # These are dimensions that have the largest absolute weights in the LASSO model
    print("Extracting top predictive dimensions...")
    dim_impacts = []
    for i, w in enumerate(tag_coeffs_adaptive):
        if abs(w) > 1e-9:
            dim_impacts.append({
                'index': i,
                'weight': float(w),
                'abs_weight': abs(float(w))
            })
    
    # Sort by absolute weight descending
    dim_impacts = sorted(dim_impacts, key=lambda x: x['abs_weight'], reverse=True)
    
    # Take top 5 or just positive if sparse (as requested)
    if len(dim_impacts) < 5:
        top_dims = [d for d in dim_impacts if d['weight'] > 0]
    else:
        top_dims = dim_impacts[:5]

    # --- SEMANTIC DIMENSIONS ANALYSIS ---
    print("Extracting top predictive semantic dimensions...")
    sem_dim_impacts = []
    for i, w in enumerate(sem_coeffs_full):
        if abs(w) > 1e-9:
            sem_dim_impacts.append({
                'index': i,
                'weight': float(w),
                'abs_weight': abs(float(w))
            })
    
    sem_dim_impacts = sorted(sem_dim_impacts, key=lambda x: x['abs_weight'], reverse=True)
    top_sem_dims = sem_dim_impacts[:5]

    # Load Semantic Dimension Labels
    SEMANTIC_LABELS_FILE = os.path.join(ROOT_DIR, "data", "production", "semantic_dimension_labels.json")
    SEMANTIC_SUM_FILE = os.path.join(ROOT_DIR, "data", "production", "semantic_sum_labels.json")
    
    sem_dimension_labels = {}
    if os.path.exists(SEMANTIC_LABELS_FILE):
        with open(SEMANTIC_LABELS_FILE, 'r') as f:
            sem_dimension_labels = json.load(f)
            
    sem_sum_labels = {}
    if os.path.exists(SEMANTIC_SUM_FILE):
        with open(SEMANTIC_SUM_FILE, 'r') as f:
            sem_sum_labels = json.load(f)
            
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
            # Handle both stringified dicts and actual dict objects
            if isinstance(tag_str, dict):
                tags_dict = tag_str
            else:
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
    supported_tag_set = set([unique_tags[i] for i in range(len(unique_tags)) if support_mask[i]])

    # --- TAG ASSOCIATION ANALYSIS (T-Test) ---
    print("Calculating tag associations (t-test)...")
    from scipy import stats
    user_df_for_tags = full_metadata.iloc[user_indices][['appid', 'tags']].copy()
    user_df_for_tags['rating'] = y[:len(user_indices)]
    
    def parse_tags_list(tag_str):
        try:
            if isinstance(tag_str, dict): return list(tag_str.keys())
            return list(ast.literal_eval(tag_str).keys())
        except: return []
    
    user_df_for_tags['tag_list'] = user_df_for_tags['tags'].apply(parse_tags_list)
    
    assoc_results = []
    for tag in supported_tag_set:
        present_mask = user_df_for_tags['tag_list'].apply(lambda x: tag in x)
        present_ratings = user_df_for_tags[present_mask]['rating'].tolist()
        absent_ratings = user_df_for_tags[~present_mask]['rating'].tolist()
        
        if len(present_ratings) < 2 or len(absent_ratings) < 2: continue
        
        t_stat, p_val = stats.ttest_ind(present_ratings, absent_ratings, equal_var=False)
        if pd.isna(p_val) or p_val >= 0.05: continue
        
        mean_diff = np.mean(present_ratings) - np.mean(absent_ratings)
        assoc_results.append({
            'tag': tag,
            'mean_diff': float(mean_diff),
            'p_value': float(p_val),
            'ratings_with': [float(r) for r in present_ratings],
            'ratings_without': [float(r) for r in absent_ratings]
        })
        
    assoc_results = sorted(assoc_results, key=lambda x: x['mean_diff'], reverse=True)
    # Filter to top/bottom 15 with strict sign check
    top_assoc = [r for r in assoc_results if r['mean_diff'] > 0][:15]
    bottom_assoc = sorted([r for r in assoc_results if r['mean_diff'] < 0], key=lambda x: x['mean_diff'])[:15]

    # Load dimension descriptions to filter their tags
    desc_path = os.path.join(ROOT_DIR, "data", "production", "tag_dimension_descriptions.json")
    dim_descriptions = {}
    if os.path.exists(desc_path):
        with open(desc_path, 'r') as f:
            dim_descriptions = json.load(f)

    # Calculate correlation data for each top dimension
    dim_correlations = {}
    dim_verified_tags = {}
    for dim in top_dims:
        idx = dim['index']
        dim_str = str(idx)
        # x is the loading of rated games on that factor (dimension)
        # y is their rating
        dim_loadings = user_tag_features_raw[:len(user_indices), idx]
        dim_correlations[dim_str] = [
            {'x': float(load), 'y': float(rat)}
            for load, rat in zip(dim_loadings, y[:len(user_indices)])
        ]

        # Sanity check tags for this dimension
        if dim_str in dim_descriptions:
            raw_pos = dim_descriptions[dim_str].get('top_positive', [])
            raw_neg = dim_descriptions[dim_str].get('top_negative', [])
            
            # Filter to only tags existing in user library
            verified_pos = [t for t in raw_pos if t in supported_tag_set]
            verified_neg = [t for t in raw_neg if t in supported_tag_set]
            
            # Fallback to global labels if no library tags match
            if not verified_pos and len(raw_pos) > 0:
                verified_pos = raw_pos[:5]
            if not verified_neg and len(raw_neg) > 0:
                verified_neg = raw_neg[:5]
            
            # Create dynamic label: A/B vs. C/D
            a = verified_pos[0] if len(verified_pos) > 0 else "?"
            b = verified_pos[1] if len(verified_pos) > 1 else ""
            c = verified_neg[0] if len(verified_neg) > 0 else "?"
            d = verified_neg[1] if len(verified_neg) > 1 else ""
            
            pos_part = f"{a}/{b}" if b else a
            neg_part = f"{c}/{d}" if d else c
            dynamic_label = f"{pos_part} vs. {neg_part}"
            
            dim_verified_tags[dim_str] = {
                'positive': verified_pos[:5],
                'negative': verified_neg[:5],
                'dynamic_label': dynamic_label
            }

    # --- SEMANTIC DIMENSIONS REFINEMENT ---
    sem_dim_correlations = {}
    sem_dim_verified_labels = {}
    
    for dim in top_sem_dims:
        idx = dim['index']
        dim_str = str(idx)
        
        # loadings are user_sem_features_scaled from earlier
        # Wait, user_sem_features_scaled was (raw / norms) * 11.25
        # We need the loadings for correlation plots
        dim_loadings = user_sem_features_scaled[:, idx]
        sem_dim_correlations[dim_str] = [
            {'x': float(load), 'y': float(rat)}
            for load, rat in zip(dim_loadings, y[:len(user_indices)])
        ]
        
        if dim_str in sem_dimension_labels:
            raw_pos = [w[0] for w in sem_dimension_labels[dim_str].get('top_positive', [])]
            raw_neg = [w[0] for w in sem_dimension_labels[dim_str].get('top_negative', [])]
            
            # Use Sum-based composite labels for the main title
            sum_label = sem_sum_labels.get(dim_str)
            if sum_label:
                dynamic_label = sum_label['dynamic_label']
            else:
                a = raw_pos[0] if len(raw_pos) > 0 else "?"
                b = raw_pos[1] if len(raw_pos) > 1 else ""
                c = raw_neg[0] if len(raw_neg) > 0 else "?"
                d = raw_neg[1] if len(raw_neg) > 1 else ""
                pos_part = f"{a}/{b}" if b else a
                neg_part = f"{c}/{d}" if d else c
                dynamic_label = f"{pos_part} vs. {neg_part}"
            
            sem_dim_verified_labels[dim_str] = {
                'positive': raw_pos[:5],
                'negative': raw_neg[:5],
                'dynamic_label': dynamic_label
            }

    # Calculate Tag Norm and Unit Vector for the Recommender (using padded full vector)
    tag_norm = np.linalg.norm(tag_coeffs_full)
    
    if tag_norm > 1e-9:
        vibe_vector_unit = (tag_coeffs_full / tag_norm).tolist()
    else:
        vibe_vector_unit = tag_coeffs_full.tolist()

    # Calculate Semantic Norm and Unit Vector
    sem_norm = np.linalg.norm(sem_coeffs_full)
    if sem_norm > 1e-9:
        sem_vibe_vector_unit = (sem_coeffs_full / sem_norm).tolist()
    else:
        sem_vibe_vector_unit = sem_coeffs_full.tolist()

    # Load whitening matrix W (original_tags x whitened_dim)
    W = np.load(W_TAG_FILE)
    
    # Project: beta_original = W * beta_whitened
    # Using the full padded vector ensures we multiply by the correct columns of W
    tag_weights_original = np.dot(W, tag_coeffs_full)
    
    if len(unique_tags) != len(tag_weights_original):
        print(f"Warning: Tag count mismatch! Names: {len(unique_tags)}, Weights: {len(tag_weights_original)}")
        # Truncate or pad to match (shouldn't happen if dataset is synced)
        min_len = min(len(unique_tags), len(tag_weights_original))
        unique_tags = unique_tags[:min_len]
        tag_weights_original = tag_weights_original[:min_len]

    # Filter the impacts: only tags with sufficient support are eligible for top/bottom lists
    eligible_impacts = [
        (t, float(w)) for i, (t, w) in enumerate(zip(unique_tags, tag_weights_original)) 
        if support_mask[i]
    ]
    
    # Sort and strictly partition by sign
    top_tags = [{'tag': t, 'impact': w} for t, w in eligible_impacts if w > 0]
    top_tags = sorted(top_tags, key=lambda x: x['impact'], reverse=True)[:10]
    
    bottom_tags = [{'tag': t, 'impact': w} for t, w in eligible_impacts if w < 0]
    bottom_tags = sorted(bottom_tags, key=lambda x: x['impact'])[:10]

    # Load All Tag Vectors for scoring and similarity
    all_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    
    # Exclude games already in user's library or manual additions (including ignored ones)
    known_indices = [appid_to_idx[aid] for aid in all_library_appids if aid in appid_to_idx]

    # --- NORTH STAR & ABYSSAL GAMES ---
    # Find games whose TASTE ALIGNMENT (Weighted Tag + Weighted Semantic) is highest/lowest.
    # This ignores metadata like quality, age, and price to find games that purely match the 'vibe'.
    print("Finding North Star and Abyssal games (Hybrid Taste Alignment)...")
    
    # 1. Calculate Tag Alignment
    all_tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r').astype(np.float32)
    # Scaled Tag Features: (Raw / Norm) * Global_Scaling
    # Alignment: Scaled @ Coeffs
    tag_alignment = (np.dot(all_vectors.astype(np.float32), tag_coeffs_full) / (all_tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    # 2. Calculate Semantic Alignment
    sem_alignment = np.zeros(len(full_metadata), dtype=np.float32)
    if sem_norm > 1e-9:
        sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
        sem_norms_all = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').astype(np.float32)
        
        batch_size = 50000
        for i in range(0, len(full_metadata), batch_size):
            end = min(i + batch_size, len(full_metadata))
            batch_vecs = sem_vectors[i:end].astype(np.float32)
            # Use same scaling as training
            batch_scaled = (batch_vecs / (sem_norms_all[i:end].reshape(-1, 1) + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
            sem_alignment[i:end] = np.dot(batch_scaled, sem_coeffs_full)

    # 3. Hybrid Alignment
    hybrid_alignment = tag_alignment + sem_alignment
    
    # Mask known games (set to a very low value so they don't appear in top/bottom)
    hybrid_alignment[known_indices] = -np.inf # Use -inf to ensure they are never picked for top

    # North Stars (Highest Alignment)
    ns_indices = np.argsort(-hybrid_alignment)[:5]
    north_stars = full_metadata.iloc[ns_indices][['appid', 'name']].copy()
    north_stars['alignment'] = [float(a) for a in hybrid_alignment[ns_indices]] # Store raw values

    # Abyssal Games (Lowest Alignment)
    hybrid_alignment[known_indices] = np.inf # Use inf to ensure they are never picked for bottom
    ab_indices = np.argsort(hybrid_alignment)[:5]
    abyssal_games = full_metadata.iloc[ab_indices][['appid', 'name']].copy()
    abyssal_games['alignment'] = [float(a) for a in hybrid_alignment[ab_indices]] # Store raw values

    # --- TOP & BOTTOM RECOMMENDATIONS ---
    print("Generating top and bottom recommendations based on solved profile (original scale)...")
    from common.utils import calculate_linear_scores
    
    # Construction coefficients at ORIGINAL scale for preview accuracy (matching ground truth 1-10)
    # Note: In build 41, we use 5.0 as the global neutral intercept for both systems.
    preview_weights = {
        'quality': float(coeffs[0]),
        'age': float(coeffs[1]),
        'popularity': float(coeffs[2]),
        'length': float(coeffs[3]),
        'difficulty': float(coeffs[4]),
        'price': float(coeffs[5]),
        'tag_match': float(tag_norm)
    }
    
    # Calculate scores on the ORIGINAL scale
    all_tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    
    # Load semantic similarities for the full dataset
    all_semantic_sims = np.zeros(len(full_metadata))
    if sem_norm > 1e-9:
        sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
        sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').reshape(-1, 1).astype(np.float32)
        
        # Unit vector for semantic coefficients
        sem_vibe_unit = sem_coeffs_full / sem_norm
        
        batch_size = 50000
        for i in range(0, len(full_metadata), batch_size):
            end = min(i + batch_size, len(full_metadata))
            batch_vecs = sem_vectors[i:end].astype(np.float32)
            # Use original penalized scaling
            batch_scaled = (batch_vecs / (sem_norms[i:end] + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
            all_semantic_sims[i:end] = np.dot(batch_scaled, sem_vibe_unit)

    scores = calculate_linear_scores(
        z_quality=quality_grid[best_idx],
        z_date=full_metadata['date_z'].values,
        z_pop=full_metadata['pop_z'].values,
        z_playtime=full_metadata['playtime_z'].values,
        z_difficulty=full_metadata['difficulty_z'].values,
        z_price=full_metadata['price_z'].values,
        tag_vectors=all_vectors,
        tag_norms=all_tag_norms,
        beta_tag=vibe_vector_unit, # Unit vector
        weights=preview_weights, # Contains 'tag_match' norm
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR,
        dot_product_lambda=DOT_PRODUCT_LAMBDA,
        z_semantic=all_semantic_sims,
        w_semantic=float(sem_norm),
        z_clamp_min=Z_SCORE_CLAMP_MIN,
        z_clamp_max=Z_SCORE_CLAMP_MAX,
        intercept=5.0 # Anchored to global prior
    )

    # --- APPLY DEFAULT FILTERS (Match Recommender) ---
    mask = np.ones(len(full_metadata), dtype=bool)
    # 1. English Only
    if 'is_english' in full_metadata.columns:
        mask &= full_metadata['is_english'].values.astype(bool)
    # 2. No VR Only
    if 'is_vr_only' in full_metadata.columns:
        mask &= ~full_metadata['is_vr_only'].values.astype(bool)
    # 3. No Utilities
    if 'is_utility' in full_metadata.columns:
        mask &= ~full_metadata['is_utility'].values.astype(bool)
    # 4. Released Only
    if 'parsed_date' in full_metadata.columns:
        build_time = pd.Timestamp.now()
        future_mask = (full_metadata['parsed_date'] > build_time).fillna(False)
        mask &= ~future_mask.values.astype(bool)
    # 5. No Delisted
    if 'is_delisted' in full_metadata.columns:
        mask &= ~full_metadata['is_delisted'].values.astype(bool)
    # 6. No Hollow Games (Metadata-deficient)
    if 'is_hollow' in full_metadata.columns:
        mask &= ~full_metadata['is_hollow'].values.astype(bool)

    # Use raw scores for sorting to maintain perfect ordinal parity with backend
    # (Clamping to 0-10 is only for display)
    sort_scores = scores.copy()
    
    # Mask known games
    sort_scores[known_indices] = -1e12
    # Apply filters
    sort_scores[~mask] = -1e12
    
    # Get top 30 using stable lexicographical sort (score DESC, name ASC)
    all_names = full_metadata['name'].fillna("").values
    top_indices = np.lexsort((all_names, -sort_scores))[:30]
    
    top_games = full_metadata.iloc[top_indices][['appid', 'name']].copy()
    # Clamp for display
    top_games['predicted_rating'] = np.clip(scores[top_indices], 0, 10)
    
    # For bottom recs, use inverse mask
    bottom_sort_scores = scores.copy()
    bottom_sort_scores[known_indices] = 1e12
    bottom_sort_scores[~mask] = 1e12 # Still exclude invalid games
    bottom_indices = np.lexsort((all_names, bottom_sort_scores))[:30]
    
    bottom_games = full_metadata.iloc[bottom_indices][['appid', 'name']].copy()
    bottom_games['predicted_rating'] = np.clip(scores[bottom_indices], 0, 10)

    # --- WEIGHT SCALING (UI SLIDERS ONLY) ---
    # We scale metadata weights so the largest absolute value is DNA_UI_SCALING_FACTOR.
    weights_to_scale = {
        'quality': float(coeffs[0]),
        'age': float(coeffs[1]),
        'popularity': float(coeffs[2]),
        'length': float(coeffs[3]),
        'difficulty': float(coeffs[4]),
        'price': float(coeffs[5]),
        'tag_match': float(tag_norm),
        'semantic': float(sem_norm) 
    }
    
    max_abs_weight = max(abs(v) for v in weights_to_scale.values())
    scaling_factor = DNA_UI_SCALING_FACTOR / max_abs_weight if max_abs_weight > 1e-6 else 1.0
    
    scaled_metadata = {k: v * scaling_factor for k, v in weights_to_scale.items()}
    scaled_metadata['discovery'] = float(optimal_disc_pref)
    
    # --- EXPLAINABILITY DATA ---
    # We want raw values for X-axis and user ratings for Y-axis
    # features: quality (at optimal disc), date_z, pop_z, playtime_z, difficulty_z, price_z
    explain_data = []
    
    # Get raw values for metadata (non-z-scored where possible, but z-scores are already normalized population-wide)
    # Actually user requested RAW values.
    # For date: release_year
    # For popularity: positive + negative
    # For length: estimated_playtime
    # For quality: quality_grid[best_idx] (This is the probit quality)
    # For price: price (need to parse from string if it's there)
    
    # Re-load metadata with raw columns
    raw_meta = pd.read_parquet(METADATA_FILE, columns=['appid', 'release_year', 'positive', 'negative', 'estimated_playtime', 'difficulty_predicted', 'price'])
    # Map price string to float
    def parse_price(p):
        if pd.isna(p) or p == '' or 'Free' in p: return 0.0
        try:
            return float(re.sub(r'[^\d.]', '', p))
        except:
            return 0.0
    import re
    raw_meta['price_raw'] = raw_meta['price'].apply(parse_price)
    
    # Slice to user games
    user_raw = raw_meta.iloc[user_indices].copy()
    user_raw['quality_raw'] = q_global # Already sliced to user_indices
    user_raw['rating'] = y[:len(user_indices)] # Exclude dummy
    
    correlations_data = {
        'quality': user_raw[['quality_raw', 'rating']].rename(columns={'quality_raw': 'x', 'rating': 'y'}).to_dict(orient='records'),
        'age': user_raw[['release_year', 'rating']].rename(columns={'release_year': 'x', 'rating': 'y'}).to_dict(orient='records'),
        'popularity': (user_raw['positive'] + user_raw['negative']).to_frame('x').assign(y=user_raw['rating']).to_dict(orient='records'),
        'length': user_raw[['estimated_playtime', 'rating']].rename(columns={'estimated_playtime': 'x', 'rating': 'y'}).to_dict(orient='records'),
        'difficulty': user_raw[['difficulty_predicted', 'rating']].rename(columns={'difficulty_predicted': 'x', 'rating': 'y'}).to_dict(orient='records'),
        'price': user_raw[['price_raw', 'rating']].rename(columns={'price_raw': 'x', 'rating': 'y'}).to_dict(orient='records'),
        'discovery': discovery_scan
    }

    # Prepare final profile
    weights = {
        'metadata': scaled_metadata,
        'correlations': correlations_data,
        'tag_dimensions': {
            'top_dims': top_dims,
            'correlations': dim_correlations,
            'verified_tags': dim_verified_tags
        },
        'semantic_dimensions': {
            'top_dims': top_sem_dims,
            'correlations': sem_dim_correlations,
            'labels': sem_dim_verified_labels
        },
        'vibe_vector': vibe_vector_unit,
        'semantic_vibe_vector': sem_vibe_vector_unit,
        'alpha': float(model.alpha_),
        'intercept': 5.0, # Match the Neutral Anchor used in preview and UI
        'scaling_factor': float(scaling_factor),
        'r2': float(r2_train), # Use the high-precision training R2
        'library_size': int(num_ratings),
        'top_tags': top_tags,
        'bottom_tags': bottom_tags,
        'associative_tags': {
            'top': top_assoc,
            'bottom': bottom_assoc
        },
        'north_stars': north_stars.to_dict(orient='records'),
        'abyssal_games': abyssal_games.to_dict(orient='records'),
        'top_recommendations': top_games.to_dict(orient='records'),
        'bottom_recommendations': bottom_games.to_dict(orient='records'),
        'ignored_appids': list(map(int, ignored_appids)),
        'library_appids': [int(aid) for aid in all_library_appids],
        'rated_appids': [int(aid) for aid in df['appid'].tolist()],
        'library_details': library_details
    }

    # Clean NaN values and NumPy types for JSON safety
    def clean_json_types(obj):
        if isinstance(obj, dict):
            return {k: clean_json_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json_types(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj

    weights = clean_json_types(weights)
    
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
