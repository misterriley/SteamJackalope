import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
import os
import sys
import json
import ast
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    PRODUCTION_DATA_DIR,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    TAG_NAMES_FILE,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR,
    TOPIC_DISTRIBUTIONS_FILE,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX,
    DIFFICULTY_NEUTRAL_FALLBACK,
    TOPIC_GLOBAL_SCALING_FACTOR
)
from common.utils import calculate_jackalope_kernel, softmin_blend, calculate_linear_scores, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS

def solve_user_taste(ground_truth_path, output_path=None):
    """
    Solves for user preference weights and generates a complete Taste DNA profile with recommendations.
    """
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    sl_path = ground_truth_path.replace('_ground_truth.csv', '_soft_labels.csv')

    steam_library_appids = set()
    library_details = {}
    if os.path.exists(sl_path):
        df_sl = pd.read_csv(sl_path)
        steam_library_appids.update(df_sl['appid'].unique().tolist())
        for _, row in df_sl.iterrows():
            aid = int(row['appid'])
            library_details[aid] = {'playtime': float(row['playtime_forever']), 'p_plus_t': float(row['p_plus_t'])}

    discovery_exclude_appids = steam_library_appids.copy()
    if 'status' in df_gt.columns:
        discovery_exclude_appids.update(df_gt[df_gt['status'].isin(['backlog', 'played', 'rated'])]['appid'].tolist())
        discovery_exclude_appids.update(df_gt[df_gt['status'] == 'ignored']['appid'].tolist())

    df = df_gt[df_gt['status'] == 'rated'].dropna(subset=['actual_rating']).copy()
    user_appids = df['appid'].values
    y = df['actual_rating'].values

    if len(df) < 5:
        print("Not enough rated games to solve Taste DNA.")
        return None

    print(f"Loading metadata and vectors...")
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}

    valid_mask = [aid in appid_to_idx for aid in user_appids]
    user_appids = user_appids[valid_mask]
    y = y[valid_mask]
    user_indices = [appid_to_idx[aid] for aid in user_appids]

    user_meta_df = full_metadata.iloc[user_indices].copy()
    
    verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r')[user_indices].astype(np.float32)
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

    # 3. Calculate Structural Archetype Features (MIGs, Meta)
    # Instead of NxN kernel (Overfit), we use alignment with Mechanical Identity Groups
    N = len(user_indices)
    
    # A. MIG Features: Game's binary membership in each MIG
    X_mig = np.zeros((N, len(MIGS)))
    mig_names = list(MIGS.keys())
    
    # Pre-calculate anchor masks for the user's rated games (Local pool)
    user_anchor_masks = {}
    all_needed_tags = set()
    for tags in MIGS.values(): all_needed_tags.update(tags)
    all_needed_tags.update(NARRATIVE_TAGS)
    all_needed_tags.update(HORROR_MARKERS)
    all_needed_tags.update(HARD_ANCHORS)
    
    tag_series = user_meta_df['tags'].fillna('').astype(str)
    for tag in all_needed_tags:
        pattern = rf"'{re.escape(tag)}':"
        user_anchor_masks[tag] = tag_series.str.contains(pattern, regex=True).values

    for j, (group, tags) in enumerate(MIGS.items()):
        m = np.zeros(N, dtype=bool)
        for t in tags:
            if t in user_anchor_masks: m |= user_anchor_masks[t]
        X_mig[:, j] = m.astype(float)

    # 4. Regression
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_all = [np.corrcoef(np.clip(quality_grid[k][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX), y)[0, 1] if np.std(quality_grid[k][user_indices]) > 1e-9 else 0 for k in range(quality_grid.shape[0])]
    best_q_idx = np.argmax(np.abs(q_all))
    q_global = np.clip(quality_grid[best_q_idx][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z', 'tone_z']
    X_meta = np.clip(user_meta_df[meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # COMBINE: X = [Q, Meta, MIGs]
    # We omit NxN kernel to prevent overfitting (R^2 > 0.7 issue)
    X = np.hstack([q_global.reshape(-1, 1), X_meta, X_mig])
    
    print(f"Solving Archetypal Ridge for {N} samples with {X.shape[1]} features...")
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0]).fit(X, y)
    r2_train = model.score(X, y)
    print(f"Model Training R^2: {r2_train:.4f}")

    # --- RESULT GENERATION ---
    q_coeff = model.coef_[0]
    meta_coeffs = model.coef_[1:7]
    mig_coeffs = model.coef_[7:]
    
    # Map MIG coefficients to meaningful impacts
    active_migs = sorted([{'group': mig_names[i], 'impact': float(mig_coeffs[i])} for i in range(len(mig_names)) if abs(mig_coeffs[i]) > 1e-4], key=lambda x: x['impact'], reverse=True)
    
    # We use active_migs as our 'Structural DNA'
    kernel_anchors = active_migs # For schema compatibility with frontend
    
    # Use top weighted MIG to find the "Best Anchor" from user's library for UI seeding
    top_mig_idx = np.argmax(mig_coeffs)
    matches_top_mig = X_mig[:, top_mig_idx] > 0.5
    if np.any(matches_top_mig):
        # Pick highest rated game from the top MIG
        best_anchor_idx = user_indices[np.where(matches_top_mig)[0][np.argmax(y[matches_top_mig])]]
    else:
        # Fallback to absolute highest rated game
        best_anchor_idx = user_indices[np.argmax(y)]

    # Full population artifacts for similarity preview
    all_verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r')
    all_sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    all_sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    all_topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    all_tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    all_tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    all_tone_z = full_metadata['tone_z'].values
    all_difficulty_z = full_metadata['difficulty_z'].values
    
    # Pre-calculate anchor masks for the entire library
    print("Pre-calculating anchor masks for full library...")
    all_anchor_masks = {}
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for tag in all_needed_tags:
        pattern = rf"'{re.escape(tag)}':"
        all_anchor_masks[tag] = tag_series_full.str.contains(pattern, regex=True).values

    # Generate Rankings for Preview
    print("Generating full library rankings for preview...")
    # Seed = Best Anchor
    tags_s_str = full_metadata.iloc[best_anchor_idx]['tags']
    tags_s_dict = ast.literal_eval(tags_s_str)
    max_s = max(tags_s_dict.values()) if tags_s_dict else 1.0
    s_tags_strict = {t for t, v in tags_s_dict.items() if v / max_s > 0.25}
    s_tags_soul = {t for t, v in tags_s_dict.items() if v / max_s > 0.15}
    s_migs = {group for group, tags in MIGS.items() if any(t in s_tags_strict for t in tags)}
    s_active_narrative = [t for t in NARRATIVE_TAGS if t in s_tags_soul]
    is_cinematic_s = "Cinematic" in s_tags_soul

    scores = calculate_jackalope_kernel(
        verb_profiles=all_verb_profiles, seed_verb_profile=all_verb_profiles[best_anchor_idx],
        sem_vectors=all_sem_vectors, sem_norms=all_sem_norms, seed_sem_vec=all_sem_vectors[best_anchor_idx], seed_sem_norm=all_sem_norms[best_anchor_idx],
        topic_distributions=all_topic_dist, seed_topic_dist=all_topic_dist[best_anchor_idx],
        topic_means=t_means, topic_stds=t_stds,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
        mature_content_flags=full_metadata['mature_content'].values > 0,
        seed_mature_content=bool(full_metadata.iloc[best_anchor_idx]['mature_content'] > 0),
        seed_migs=s_migs, seed_tags=s_tags_soul,
        candidate_anchor_masks=all_anchor_masks,
        active_narrative_seed=s_active_narrative,
        is_cinematic_seed=is_cinematic_s,
        difficulty_z=all_difficulty_z, seed_difficulty_z=all_difficulty_z[best_anchor_idx],
        tone_z=all_tone_z, seed_tone_z=all_tone_z[best_anchor_idx]
    )

    # Apply Metadata Coefficients
    scores = (scores * 5.0) + model.intercept_
    scores += (np.clip(quality_grid[best_q_idx], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) * q_coeff)
    scores += (np.clip(full_metadata['date_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) * meta_coeffs[0])
    scores += (np.clip(full_metadata['pop_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) * meta_coeffs[1])
    scores += (np.clip(full_metadata['playtime_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) * meta_coeffs[2])
    scores += (np.clip(all_difficulty_z, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) * meta_coeffs[3])
    scores += (np.clip(full_metadata['price_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) * meta_coeffs[4])
    scores += (np.clip(all_tone_z, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) * meta_coeffs[5])

    # Filters
    mask = np.ones(len(full_metadata), dtype=bool)
    mask &= full_metadata['is_english'].values.astype(bool)
    mask &= ~full_metadata['is_vr_only'].values.astype(bool)
    mask &= ~full_metadata['is_utility'].values.astype(bool)
    mask &= ~full_metadata['is_delisted'].values.astype(bool)
    mask &= ~full_metadata['is_hollow'].values.astype(bool)

    # Exclude indices
    completed_indices = [appid_to_idx[aid] for aid in discovery_exclude_appids if aid in appid_to_idx]
    
    # Sort for Recommendations
    sort_scores = scores.copy()
    sort_scores[completed_indices] = -1e12
    sort_scores[~mask] = -1e12
    top_indices = np.argsort(-sort_scores)[:30]
    top_games = full_metadata.iloc[top_indices][['appid', 'name']].copy()
    top_games['predicted_rating'] = np.clip(scores[top_indices], 0, 10)

    bottom_sort_scores = scores.copy()
    bottom_sort_scores[completed_indices] = 1e12
    bottom_sort_scores[~mask] = 1e12
    bottom_indices = np.argsort(bottom_sort_scores)[:30]
    bottom_games = full_metadata.iloc[bottom_indices][['appid', 'name']].copy()
    bottom_games['predicted_rating'] = np.clip(scores[bottom_indices], 0, 10)

    # Backlog Recs
    owned_but_unplayed = [aid for aid, d in library_details.items() if d['playtime'] == 0 and aid in appid_to_idx]
    backlog_indices = [appid_to_idx[aid] for aid in owned_but_unplayed if aid in appid_to_idx]
    backlog_recs = pd.DataFrame()
    if backlog_indices:
        b_scores = scores[backlog_indices]
        b_top = np.argsort(-b_scores)[:30]
        backlog_recs = full_metadata.iloc[np.array(backlog_indices)[b_top]][['appid', 'name']].copy()
        backlog_recs['predicted_rating'] = np.clip(b_scores[b_top], 0, 10)

    # North Stars (Highest Pure Vibe Alignment)
    # Re-calculate kernel with neutral meta to find pure soulmates
    vibe_only_scores = calculate_jackalope_kernel(
        verb_profiles=all_verb_profiles, seed_verb_profile=all_verb_profiles[best_anchor_idx],
        sem_vectors=all_sem_vectors, sem_norms=all_sem_norms, seed_sem_vec=all_sem_vectors[best_anchor_idx], seed_sem_norm=all_sem_norms[best_anchor_idx],
        topic_distributions=all_topic_dist, seed_topic_dist=all_topic_dist[best_anchor_idx],
        topic_means=t_means, topic_stds=t_stds,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
        mature_content_flags=full_metadata['mature_content'].values > 0,
        seed_mature_content=bool(full_metadata.iloc[best_anchor_idx]['mature_content'] > 0),
        seed_migs=s_migs, seed_tags=s_tags_soul,
        candidate_anchor_masks=all_anchor_masks,
        active_narrative_seed=s_active_narrative,
        is_cinematic_seed=is_cinematic_s,
        difficulty_z=all_difficulty_z, seed_difficulty_z=all_difficulty_z[best_anchor_idx],
        tone_z=all_tone_z, seed_tone_z=all_tone_z[best_anchor_idx]
    )
    vibe_only_scores[completed_indices] = -1e12
    vibe_only_scores[~mask] = -1e12
    ns_indices = np.argsort(-vibe_only_scores)[:5]
    north_stars = full_metadata.iloc[ns_indices][['appid', 'name']].copy()
    north_stars['alignment'] = vibe_only_scores[ns_indices]

    # Associative Tags (Regression on Tags)
    print("Calculating associative tag impacts...")
    # This is a bit heavy, so we subsample or use a simpler model
    from sklearn.linear_model import Lasso
    tag_X = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices]
    tag_model = Lasso(alpha=0.1).fit(tag_X, y)
    tag_names = json.load(open(TAG_NAMES_FILE, 'r'))
    tag_impacts = sorted([{'tag': tag_names[i], 'impact': float(w)} for i, w in enumerate(tag_model.coef_) if abs(w) > 1e-4], key=lambda x: x['impact'], reverse=True)
    
    # Favorite-Seed Recommendations
    favorites_df = df[df['actual_rating'] >= 9].copy()
    favorite_recs = []
    for _, fav in favorites_df.head(10).iterrows():
        f_idx = appid_to_idx[int(fav['appid'])]
        
        # PROPERLY EXTRACT SEED METADATA FOR THIS SPECIFIC FAVORITE
        tags_f_str = full_metadata.iloc[f_idx]['tags']
        tags_f_dict = ast.literal_eval(tags_f_str)
        max_f = max(tags_f_dict.values()) if tags_f_dict else 1.0
        f_tags_strict = {t for t, v in tags_f_dict.items() if v / max_f > 0.35}
        f_tags_soul = {t for t, v in tags_f_dict.items() if v / max_f > 0.15}
        f_migs = {group for group, tags in MIGS.items() if any(t in f_tags_strict for t in tags)}
        f_active_narrative = [t for t in NARRATIVE_TAGS if t in f_tags_soul]
        is_cinematic_f = "Cinematic" in f_tags_soul

        f_scores = calculate_jackalope_kernel(
            verb_profiles=all_verb_profiles, seed_verb_profile=all_verb_profiles[f_idx],
            sem_vectors=all_sem_vectors, sem_norms=all_sem_norms, seed_sem_vec=all_sem_vectors[f_idx], seed_sem_norm=all_sem_norms[f_idx],
            topic_distributions=all_topic_dist, seed_topic_dist=all_topic_dist[f_idx],
            topic_means=t_means, topic_stds=t_stds,
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
            sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
            mature_content_flags=full_metadata['mature_content'].values > 0,
            seed_mature_content=bool(full_metadata.iloc[f_idx]['mature_content'] > 0),
            seed_migs=f_migs, seed_tags=f_tags_soul,
            candidate_anchor_masks=all_anchor_masks,
            active_narrative_seed=f_active_narrative,
            is_cinematic_seed=is_cinematic_f,
            difficulty_z=all_difficulty_z, seed_difficulty_z=all_difficulty_z[f_idx],
            tone_z=all_tone_z, seed_tone_z=all_tone_z[f_idx]
        )
        f_scores[completed_indices] = -1e12
        f_scores[~mask] = -1e12
        f_top = np.argsort(-f_scores)[:3]
        favorite_recs.append({
            'seed_appid': int(fav['appid']),
            'seed_name': str(fav['name']),
            'top_games': full_metadata.iloc[f_top][['appid', 'name']].assign(predicted_rating=np.clip(f_scores[f_top]*10,0,10)).to_dict(orient='records')
        })

    # Prepare Final JSON
    result = {
        'metadata': {
            'quality': float(q_coeff), 'age': float(meta_coeffs[0]), 'popularity': float(meta_coeffs[1]), 
            'length': float(meta_coeffs[2]), 'difficulty': float(meta_coeffs[3]), 'price': float(meta_coeffs[4]),
            'tone': float(meta_coeffs[5]), 'tag_match': 1.0, 'semantic': 1.0, 'topic_match': 0.1
        },
        'kernel_anchors': kernel_anchors[:50], 'r2': float(r2_train), 
        'vibe_vector': (all_tag_vectors[best_anchor_idx] / (all_tag_norms[best_anchor_idx] + 1e-9)).tolist(),
        'semantic_vibe_vector': (all_sem_vectors[best_anchor_idx] / (all_sem_norms[best_anchor_idx] + 1e-9)).tolist(),
        'topic_vibe_vector': all_topic_dist[best_anchor_idx].tolist(),
        'intercept': float(model.intercept_), 'scaling_factor': 3.0, 
        'library_appids': [int(aid) for aid in discovery_exclude_appids],
        'rated_appids': [int(aid) for aid in user_appids], 'library_size': len(full_metadata),
        'top_recommendations': top_games.to_dict(orient='records'),
        'bottom_recommendations': bottom_games.to_dict(orient='records'),
        'backlog_recommendations': backlog_recs.to_dict(orient='records'),
        'north_stars': north_stars.to_dict(orient='records'),
        'associative_tags': {
            'top': tag_impacts[:10],
            'bottom': tag_impacts[-10:]
        },
        'favorite_game_recommendations': favorite_recs
    }
    
    if output_path:
        with open(output_path, 'w') as f: json.dump(result, f, indent=4)
    return result

if __name__ == "__main__":
    sid = sys.argv[1]
    solve_user_taste(f"data/user_{sid}_ground_truth.csv", output_path=f"data/user_{sid}_taste_profile.json")
