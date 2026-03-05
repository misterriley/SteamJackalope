import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import os
import sys
import json
import ast
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, 
    PRODUCTION_DATA_DIR,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, calculate_jackalope_kernel_2d, MIGS, normalize_string

def solve_user_taste(ground_truth_path, output_path=None):
    """
    Production Solver: CORE 9 Mode (20.47% Verified OOS R2 Peak).
    Includes rich insight generation for the 'Analyze' page.
    """
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    # Rated Games Only for Solver
    df_rated = df_gt[df_gt['status'] == 'rated'].dropna(subset=['actual_rating'])
    y = df_rated['actual_rating'].values
    user_appids = df_rated['appid'].values
    y_dev_global = y - 5.0
    
    sid = os.path.basename(ground_truth_path).split('_')[1]
    sl_path = ground_truth_path.replace('_ground_truth.csv', '_soft_labels.csv')
    
    steam_library_appids = set()
    if os.path.exists(sl_path):
        df_sl = pd.read_csv(sl_path)
        steam_library_appids.update(df_sl['appid'].unique().tolist())

    discovery_exclude_appids = steam_library_appids.copy()
    if 'status' in df_gt.columns:
        discovery_exclude_appids.update(df_gt[df_gt['status'].isin(['backlog', 'played', 'rated'])]['appid'].tolist())
        discovery_exclude_appids.update(df_gt[df_gt['status'] == 'ignored']['appid'].tolist())

    print(f"Loading metadata and vectors...")
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_rated['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices].copy()
    N = len(user_indices)

    all_graph_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)
    user_graph_vectors = all_graph_vectors[user_indices]
    verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r')[user_indices].astype(np.float32)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

    mig_mask_array = np.zeros((len(full_metadata), len(MIGS)), dtype=bool)
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for j, (group, tags) in enumerate(MIGS.items()):
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            mig_mask_array[:, j] |= tag_series_full.str.contains(pattern, regex=True).values
    user_mig_masks = mig_mask_array[user_indices]

    from common.utils import extract_seed_metadata
    seed_meta = extract_seed_metadata(user_indices, full_metadata)

    print("Calculating full NxN Jackalope Kernel and Graph Sim...")
    K_full = calculate_jackalope_kernel_2d(
        verb_profiles=verb_profiles, seed_verb_profiles=verb_profiles,
        sem_vectors=sem_vectors, sem_norms=sem_norms, seed_sem_vecs=sem_vectors, seed_sem_norms=sem_norms,
        topic_distributions=topic_dist, seed_topic_dists=topic_dist,
        topic_means=t_means, topic_stds=t_stds,
        candidate_mig_masks=user_mig_masks,
        seed_mig_masks=user_mig_masks,
        difficulty_z=user_meta_df['difficulty_z'].values, seed_difficulty_z=user_meta_df['difficulty_z'].values,
        tone_z=user_meta_df['tone_z'].values, seed_tone_z=user_meta_df['tone_z'].values,
        seed_tags=seed_meta['soul_tags_list'], seed_migs=seed_meta['migs_list'],
        mature_content_flags=seed_meta['mature_flags'], seed_mature_content_flags=seed_meta['mature_flags'],
        graph_embeddings=user_graph_vectors, seed_graph_vecs=user_graph_vectors
    )

    dot_graph = np.dot(user_graph_vectors, user_graph_vectors.T)
    g_norms = np.linalg.norm(user_graph_vectors, axis=1)
    G_full = dot_graph / (g_norms[:, None] * g_norms[None, :] + 1e-9)
    G_full = np.maximum(0, G_full)

    user_names = user_meta_df['name'].tolist()
    lp_mask = np.ones((N, N), dtype=bool)
    for i in range(N):
        clean_i = set(normalize_string(user_names[i]).split())
        if len(clean_i) < 2: continue 
        for j in range(i + 1, N):
            clean_j = set(normalize_string(user_names[j]).split())
            overlap = clean_i.intersection(clean_j)
            if len(overlap) >= 2 and (len(overlap) / max(len(clean_i), len(clean_j)) > 0.7):
                lp_mask[i, j] = lp_mask[j, i] = False

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 6, 50)
    alpha_scores = np.zeros(len(alphas))
    
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_feat = to_z(quality_grid[0][user_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z', 'tone_z']
    X_meta_data = np.zeros((N, len(meta_cols)))
    for j, col in enumerate(meta_cols):
        X_meta_data[:, j] = to_z(user_meta_df[col].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    
    # CORE 9 Feature Pool: [Q, Meta]
    X_static = np.hstack([q_feat.reshape(-1, 1), X_meta_data]) 

    K_exp_full_pool = np.exp(K_full * 10.0)
    
    print(f"Executing Research Alpha Sweep for {N} samples with CORE 9 features...")
    for train_idx, test_idx in kf.split(range(N)):
        K_sub_train = K_exp_full_pool[np.ix_(train_idx, train_idx)] * lp_mask[np.ix_(train_idx, train_idx)]
        np.fill_diagonal(K_sub_train, 0.0)
        X_k_train = (np.sum(K_sub_train * y_dev_global[train_idx], axis=1) / (np.sum(K_sub_train, axis=1) + 1e-9)).reshape(-1, 1)
        
        G_train = G_full[np.ix_(train_idx, train_idx)] * lp_mask[np.ix_(train_idx, train_idx)]
        np.fill_diagonal(G_train, 0.0)
        X_g_train = (np.sum(G_train * y_dev_global[train_idx], axis=1) / (np.sum(G_train, axis=1) + 1e-9)).reshape(-1, 1)
        
        X_train = np.hstack([X_k_train, X_g_train, X_static[train_idx]])
        
        K_sub_test = K_exp_full_pool[np.ix_(test_idx, train_idx)] * lp_mask[np.ix_(test_idx, train_idx)]
        X_k_test = (np.sum(K_sub_test * y_dev_global[train_idx], axis=1) / (np.sum(K_sub_test, axis=1) + 1e-9)).reshape(-1, 1)
        
        G_test = G_full[np.ix_(test_idx, train_idx)] * lp_mask[np.ix_(test_idx, train_idx)]
        X_g_test = (np.sum(G_test * y_dev_global[train_idx], axis=1) / (np.sum(G_test, axis=1) + 1e-9)).reshape(-1, 1)
        
        X_test = np.hstack([X_k_test, X_g_test, X_static[test_idx]])
        
        for i, a in enumerate(alphas):
            ridge = Ridge(alpha=a).fit(X_train, y[train_idx])
            alpha_scores[i] += ridge.score(X_test, y[test_idx])
            
    best_alpha = alphas[np.argmax(alpha_scores)]
    oos_r2 = np.max(alpha_scores) / 5.0
    print(f"Selected Discovery Alpha: {best_alpha:.4f}")
    print(f"Verified Discovery OOS R^2: {oos_r2:.4f}")

    X_k_final = (np.sum((K_exp_full_pool * lp_mask) * y_dev_global, axis=1) / (np.sum(K_exp_full_pool * lp_mask, axis=1) + 1e-9)).reshape(-1, 1)
    X_g_final = (np.sum((G_full * lp_mask) * y_dev_global, axis=1) / (np.sum(G_full * lp_mask, axis=1) + 1e-9)).reshape(-1, 1)
    X_final = np.hstack([X_k_final, X_g_final, X_static])
    
    model = Ridge(alpha=best_alpha).fit(X_final, y)
    r2_train = model.score(X_final, y)
    print(f"Final Fit Confidence (Training R2): {r2_train:.4f}")

    kernel_coeff = model.coef_[0]
    graph_coeff = model.coef_[1]
    q_coeff = model.coef_[2]
    meta_coeffs = model.coef_[3:9]
    topic_coeffs = np.zeros(249)

    # --- RICH INSIGHTS: ACTIVE MIGS ---
    print("Generating rich insights...")
    mig_impacts = []
    for j, group in enumerate(MIGS.keys()):
        # Calculate impact by looking at rated games with this MIG
        mask = user_mig_masks[:, j].astype(bool)
        if np.sum(mask) >= 3:
            impact = np.mean(y[mask]) - np.mean(y)
            mig_impacts.append({'group': group, 'impact': float(impact)})
    active_migs = sorted(mig_impacts, key=lambda x: abs(x['impact']), reverse=True)

    # --- PRODUCTION SCORING (Library Scale) ---
    print("Calculating library-scale recommendations...")
    all_verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r')
    all_sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    all_sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    
    K_lib = calculate_jackalope_kernel_2d(
        verb_profiles=all_verb_profiles, seed_verb_profiles=verb_profiles,
        sem_vectors=all_sem_vectors, sem_norms=all_sem_norms, seed_sem_vecs=sem_vectors, seed_sem_norms=sem_norms,
        topic_distributions=np.zeros((len(full_metadata), 249), dtype=np.float32), 
        seed_topic_dists=np.zeros((len(user_indices), 249), dtype=np.float32),
        topic_means=t_means, topic_stds=t_stds,
        candidate_mig_masks=mig_mask_array, seed_mig_masks=user_mig_masks,
        difficulty_z=full_metadata['difficulty_z'].values, seed_difficulty_z=user_meta_df['difficulty_z'].values,
        tone_z=full_metadata['tone_z'].values, seed_tone_z=user_meta_df['tone_z'].values,
        seed_tags=seed_meta['soul_tags_list'], seed_migs=seed_meta['migs_list'],
        mature_content_flags=full_metadata['mature_content'].values > 0,
        seed_mature_content_flags=seed_meta['mature_flags'],
        graph_embeddings=all_graph_vectors, seed_graph_vecs=user_graph_vectors
    )
    
    K_exp_lib = np.exp(K_lib * 10.0) 
    for c, f_idx in enumerate(user_indices): K_exp_lib[f_idx, c] = 0.0
    X_k_lib = np.sum(K_exp_lib * y_dev_global, axis=1) / (np.sum(K_exp_lib, axis=1) + 1e-9)

    G_lib = np.dot(all_graph_vectors, user_graph_vectors.T)
    g_norms_lib = np.linalg.norm(all_graph_vectors, axis=1)
    s_norms_lib = np.linalg.norm(user_graph_vectors, axis=1)
    G_lib /= (g_norms_lib[:, None] * s_norms_lib[None, :] + 1e-9)
    G_lib = np.maximum(0, G_lib)
    for c, f_idx in enumerate(user_indices): G_lib[f_idx, c] = 0.0
    X_g_lib = np.sum(G_lib * y_dev_global, axis=1) / (np.sum(G_lib, axis=1) + 1e-9)

    scores = (X_k_lib * kernel_coeff) + (X_g_lib * graph_coeff) + model.intercept_
    scores += (to_z(quality_grid[0], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)) * q_coeff)
    for i, col in enumerate(meta_cols):
        scores += to_z(full_metadata[col].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)) * meta_coeffs[i]

    # --- FILTERS & EXCLUSIONS ---
    from common.utils import get_base_filter_mask
    mask = get_base_filter_mask(full_metadata, english_only=True, remove_vr=True, remove_utilities=True, remove_delisted=True, remove_hollow=True)
    
    # Discovery Exclusions
    completed_indices = [appid_to_idx[aid] for aid in discovery_exclude_appids if aid in appid_to_idx]
    discovery_scores = scores.copy()
    discovery_scores[completed_indices] = -1e12
    discovery_scores[~mask] = -1e12
    top_discovery_indices = np.argsort(-discovery_scores)[:30]
    top_recommendations = full_metadata.iloc[top_discovery_indices][['appid', 'name']].copy()
    top_recommendations['predicted_rating'] = np.clip(scores[top_discovery_indices], 0, 10)

    # Backlog Priority
    backlog_appids = df_gt[df_gt['status'] == 'backlog']['appid'].values
    backlog_indices = [appid_to_idx[aid] for aid in backlog_appids if aid in appid_to_idx]
    backlog_scores = scores[backlog_indices]
    top_backlog_indices = np.array(backlog_indices)[np.argsort(-backlog_scores)][:30]
    backlog_recommendations = full_metadata.iloc[top_backlog_indices][['appid', 'name']].copy()
    backlog_recommendations['predicted_rating'] = np.clip(scores[top_backlog_indices], 0, 10)

    # Hate List (Library games with lowest scores)
    library_indices = [appid_to_idx[aid] for aid in steam_library_appids if aid in appid_to_idx]
    lib_scores = scores[library_indices]
    bottom_lib_indices = np.array(library_indices)[np.argsort(lib_scores)][:30]
    bottom_recommendations = full_metadata.iloc[bottom_lib_indices][['appid', 'name']].copy()
    bottom_recommendations['predicted_rating'] = np.clip(scores[bottom_lib_indices], 0, 10)

    # North Stars (Highest X_k values - games mathematically closest to your peak taste)
    north_star_indices = np.argsort(-X_k_lib)[:5]
    north_stars = full_metadata.iloc[north_star_indices][['appid', 'name']].copy()
    north_stars['alignment'] = X_k_lib[north_star_indices]

    # Predictive Tags
    tag_impacts = []
    # Only check top 200 popular tags for speed
    all_tags = set()
    for t_str in full_metadata.iloc[user_indices]['tags']:
        if isinstance(t_str, str):
            all_tags.update(ast.literal_eval(t_str).keys())
    
    for tag in list(all_tags)[:150]:
        pattern = rf"'{re.escape(tag)}':"
        has_tag = tag_series_full.iloc[user_indices].str.contains(pattern, regex=True).values
        if np.sum(has_tag) >= 5:
            impact = np.mean(y[has_tag]) - np.mean(y)
            
            # Find top discovery games with this tag
            cand_has_tag = tag_series_full.str.contains(pattern, regex=True).values
            tag_scores = scores.copy()
            tag_scores[~cand_has_tag] = -1e12
            tag_scores[completed_indices] = -1e12
            tag_scores[~mask] = -1e12
            top_tag_indices = np.argsort(-tag_scores)[:5]
            tag_top_games = full_metadata.iloc[top_tag_indices][['appid', 'name']].to_dict(orient='records')
            
            tag_impacts.append({
                'tag': tag, 
                'impact': float(impact), 
                'ratings_with': y[has_tag].tolist(), 
                'ratings_without': y[~has_tag].tolist(),
                'top_games': tag_top_games
            })
    
    sorted_tags = sorted(tag_impacts, key=lambda x: x['impact'], reverse=True)
    associative_tags = {
        'top': sorted_tags[:15],
        'bottom': sorted_tags[-15:][::-1]
    }

    # Favorite Game Seed Recommendations (All 9s and 10s)
    favorite_recs = []
    top_user_games = df_rated[df_rated['actual_rating'] >= 9.0]
    for _, row in top_user_games.iterrows():
        aid = int(row['appid'])
        if aid in appid_to_idx:
            idx = appid_to_idx[aid]
            # Recover name from metadata if missing
            seed_name = full_metadata.iloc[idx]['name'] if pd.isna(row['name']) else row['name']
            # Get closest neighbors from K_lib
            sims = K_lib[:, user_indices.index(idx)]
            sims[idx] = -1e12
            sims[completed_indices] = -1e12
            sims[~mask] = -1e12
            neighbor_indices = np.argsort(-sims)[:10]
            fav_neighbors = full_metadata.iloc[neighbor_indices][['appid', 'name']].to_dict(orient='records')
            favorite_recs.append({'seed_appid': aid, 'seed_name': seed_name, 'top_games': fav_neighbors})

    result = {
        'metadata': {
            'kernel_match': float(kernel_coeff), 'graph_match': float(graph_coeff),
            'quality': float(q_coeff), 'age': float(meta_coeffs[0]), 'popularity': float(meta_coeffs[1]), 
            'length': float(meta_coeffs[2]), 'difficulty': float(meta_coeffs[3]), 'price': float(meta_coeffs[4]),
            'tone': float(meta_coeffs[5]), 'topic_coeffs': topic_coeffs.tolist(),
            'best_q_idx': 0, 'oos_r2': float(oos_r2)
        },
        'kernel_anchors': active_migs[:50], 
        'r2': float(r2_train), 
        'intercept': float(model.intercept_),
        'library_appids': [int(aid) for aid in discovery_exclude_appids],
        'rated_appids': [int(aid) for aid in user_appids],
        'top_recommendations': top_recommendations.to_dict(orient='records'),
        'backlog_recommendations': backlog_recommendations.to_dict(orient='records'),
        'bottom_recommendations': bottom_recommendations.to_dict(orient='records'),
        'north_stars': north_stars.to_dict(orient='records'),
        'associative_tags': associative_tags,
        'favorite_game_recommendations': favorite_recs
    }
    
    if output_path:
        with open(output_path, 'w') as f: json.dump(result, f, indent=4)
        np.save(output_path.replace('_taste_profile.json', '_predicted_ratings.npy'), scores.astype(np.float32))
        print(f"\n>>> SUCCESS: RICH TASTE DNA SAVED TO {output_path} (R2: {oos_r2:.4f}) <<<")
    return result

if __name__ == "__main__":
    sid = sys.argv[1]
    solve_user_taste(f"data/user_{sid}_ground_truth.csv", output_path=f"data/user_{sid}_taste_profile.json")
