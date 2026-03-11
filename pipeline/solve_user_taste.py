import pandas as pd
import numpy as np
import scipy.stats
import math
import os
import sys
import json
import ast
import re
from sklearn.linear_model import Ridge

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, 
    PRODUCTION_DATA_DIR,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)
from common.utils import to_z

def normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms

def get_list(val):
    if pd.isna(val).all() if hasattr(val, 'all') else pd.isna(val): return []
    if isinstance(val, dict): return list(val.keys())
    if isinstance(val, str):
        try:
            d = eval(val)
            if isinstance(d, dict): return list(d.keys())
        except: pass
        return [x.strip() for x in val.split(',')]
    if hasattr(val, 'tolist'): return val.tolist()
    if isinstance(val, np.ndarray) and len(val.shape) == 0 and isinstance(val.item(), dict):
        return list(val.item().keys())
    return list(val)

def identify_puzzle_subgenre(tags_set):
    if 'Hidden Object' in tags_set: return 'Hidden Object'
    if 'Automation' in tags_set or 'Programming' in tags_set: return 'Automation'
    if 'Sokoban' in tags_set or 'Grid-Based Movement' in tags_set: return 'Sokoban/Grid'
    if 'Puzzle' in tags_set and ('First-Person' in tags_set or '3D Platformer' in tags_set or 'Open World' in tags_set): return 'Spatial/3D'
    return 'Generic/Other'

def calculate_subversion_probability(tags_set):
    prob = 0.0
    has_psych_horror = 'Psychological Horror' in tags_set
    has_surreal = 'Surreal' in tags_set
    has_satire = 'Satire' in tags_set
    innocent_tags = {'Cute', 'Education', 'Dating Sim', 'Family Friendly', 'Farming Sim', 'Typing', 'Math', 'Software', 'Game Development'}
    intersecting_innocent = tags_set.intersection(innocent_tags)
    if len(intersecting_innocent) > 0:
        if has_psych_horror:
            if 'Dating Sim' in intersecting_innocent: prob = max(prob, 0.95)
            elif 'Cute' in intersecting_innocent or 'Family Friendly' in intersecting_innocent: prob = max(prob, 0.85)
            else: prob = max(prob, 0.60)
        if has_satire:
            if 'Farming Sim' in intersecting_innocent or 'Game Development' in intersecting_innocent: prob = max(prob, 0.40)
            else: prob = max(prob, 0.25)
        if has_surreal:
            if 'Education' in intersecting_innocent or 'Math' in intersecting_innocent: prob = max(prob, 0.30)
            else: prob = max(prob, 0.15)
    return prob

def calc_aicc(y, X):
    X_mat = np.column_stack((np.ones(len(y)), X))
    beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    preds = X_mat @ beta
    resid = y - preds
    rss = np.sum(resid**2)
    n = len(y)
    k = X_mat.shape[1]
    aic = n * np.log(rss / n) + 2 * k
    if n - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = np.inf
    return aicc, preds, resid, beta

def golden_section_search(f, a, b, tol=0.1):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = f(c)
    fd = f(d)
    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - resphi * (b - a)
            fd = f(d)
    return (b + a) / 2

def solve_user_taste(ground_truth_path, output_path=None):
    print(f"Loading data from {ground_truth_path}...")
    df = pd.read_parquet(METADATA_FILE)
    N_all = len(df)
    
    gt = pd.read_csv(ground_truth_path)
    gt_rated = gt[gt['status'] == 'rated'].copy()
    all_gt_appids = set(gt['appid'].tolist())
    
    unplayed_statuses = ['backlog', 'unplayed', 'wishlist']
    backlog_appids = set(gt[gt['status'].isin(unplayed_statuses)]['appid'].tolist())
    if len(backlog_appids) == 0:
        backlog_appids = set(gt[~gt['status'].isin(['rated', 'ignored', 'played'])]['appid'].tolist())
        
    ignored_appids = set(gt[gt['status'] == 'ignored']['appid'].tolist())
    
    merged = gt_rated.merge(df[['appid']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    src_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    
    print("Analyzing predictive tags...")
    tag_lists_src = [set(get_list(t)) for t in df.iloc[src_idxs]['tags']]
    all_tags = set()
    for t_set in tag_lists_src: all_tags.update(t_set)
        
    tag_stats = []
    for tag in list(all_tags)[:150]:
        with_tag = []
        without_tag = []
        for i, t_set in enumerate(tag_lists_src):
            if tag in t_set: with_tag.append(actual_ratings[i])
            else: without_tag.append(actual_ratings[i])
                
        if len(with_tag) >= 5 and len(without_tag) >= 5:
            t_stat, p_val = scipy.stats.ttest_ind(with_tag, without_tag, equal_var=False)
            if p_val < 0.05 and np.mean(with_tag) > np.mean(without_tag):
                tag_stats.append({
                    'tag': tag,
                    'mean_with': np.mean(with_tag),
                    'mean_without': np.mean(without_tag),
                    'diff': np.mean(with_tag) - np.mean(without_tag),
                    'p_val': p_val,
                    'count': len(with_tag)
                })
                
    tag_stats.sort(key=lambda x: x['diff'], reverse=True)
    
    print("Loading feature matrices...")
    f_tags = normalize(np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r'))
    f_desc = normalize(np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r'))
    f_verbs = normalize(np.load(os.path.join(PRODUCTION_DATA_DIR, 'diffused_verb_profiles.npy'), mmap_mode='r').astype(np.float32))
    f_graph = normalize(np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r'))
    
    pop_z = df['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    print("Finding Best Quality Grid Level...")
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, 'quality_scores_grid.npy'), mmap_mode='r')
    best_corr, best_row = -1, -1
    for i in range(quality_grid.shape[0]):
        q = quality_grid[i, src_idxs]
        corr, _ = scipy.stats.pearsonr(q, actual_ratings)
        if corr > best_corr:
            best_corr = corr
            best_row = i
            
    quality_feature_all = quality_grid[best_row, :]
    
    print("Forward Selection of Linear Predictors (AICc)...")
    global_features = {
        'Quality': quality_feature_all,
        'Popularity': df['pop_z'].fillna(0).values,
        'Price': df['price_z'].fillna(0).values,
        'Age': df['date_z'].fillna(0).values,
        'Length': df['playtime_z'].fillna(0).values,
        'Difficulty': df['difficulty_z'].fillna(0).values,
        'Tone': df['tone_z'].fillna(0).values
    }
    
    selected_feature_names = ['Quality']
    remaining_features = ['Popularity', 'Price', 'Age', 'Length', 'Difficulty', 'Tone']
    X_train_current = global_features['Quality'][src_idxs].reshape(-1, 1)
    current_aicc, base_preds, current_residuals, current_beta = calc_aicc(actual_ratings, X_train_current)
    
    while remaining_features:
        best_candidate, best_candidate_aicc = None, current_aicc
        for candidate in remaining_features:
            X_test = np.column_stack((X_train_current, global_features[candidate][src_idxs]))
            test_aicc, _, _, _ = calc_aicc(actual_ratings, X_test)
            if test_aicc < best_candidate_aicc:
                best_candidate_aicc = test_aicc
                best_candidate = candidate
        if best_candidate:
            selected_feature_names.append(best_candidate)
            remaining_features.remove(best_candidate)
            X_train_current = np.column_stack((X_train_current, global_features[best_candidate][src_idxs]))
            current_aicc = best_candidate_aicc
        else:
            break
            
    _, _, final_residuals, final_beta = calc_aicc(actual_ratings, X_train_current)
    X_global_selected = np.column_stack([global_features[f] for f in selected_feature_names])
    X_global_mat = np.column_stack((np.ones(N_all), X_global_selected))
    base_preds_all = X_global_mat @ final_beta
    
    print("Tuning Similarity Power (Golden Section Search)...")
    sim_tags_loo = np.dot(f_tags[src_idxs], f_tags[src_idxs].T)
    sim_desc_loo = np.dot(f_desc[src_idxs], f_desc[src_idxs].T)
    sim_verbs_loo = np.dot(f_verbs[src_idxs], f_verbs[src_idxs].T)
    sim_graph_loo = np.dot(f_graph[src_idxs], f_graph[src_idxs].T) * pop_discount[src_idxs][:, None]
    
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}
    sim_matrix_loo = (
        weights['tags'] * sim_tags_loo +
        weights['desc'] * sim_desc_loo +
        weights['verbs'] * sim_verbs_loo +
        weights['graph'] * sim_graph_loo
    )
    
    tags_list_all = [set(get_list(x)) for x in df['tags']]
    subgenres_all = np.array([identify_puzzle_subgenre(t) for t in tags_list_all])
    subv_probs_all = np.array([calculate_subversion_probability(t) for t in tags_list_all])
    
    subgenres_src = subgenres_all[src_idxs]
    subv_probs_src = subv_probs_all[src_idxs]
    
    for j in range(len(src_idxs)):
        src_subg = subgenres_src[j]
        src_prob = subv_probs_src[j]
        if src_subg != 'Generic/Other':
            mask = (subgenres_src != 'Generic/Other') & (subgenres_src != src_subg)
            sim_matrix_loo[mask, j] -= 0.3
        if src_prob > 0:
            joint_probs = np.sqrt(src_prob * subv_probs_src)
            sim_matrix_loo[:, j] += (0.45 * joint_probs)
            
    np.fill_diagonal(sim_matrix_loo, 0)
    
    def objective_function(power):
        weight_matrix = np.sign(sim_matrix_loo) * (np.abs(sim_matrix_loo) ** power)
        sum_abs_w = np.sum(np.abs(weight_matrix), axis=1)
        valid_mask = sum_abs_w > 0
        loo_preds = np.zeros(len(final_residuals))
        if np.any(valid_mask):
            loo_preds[valid_mask] = np.sum(weight_matrix[valid_mask] * final_residuals, axis=1) / sum_abs_w[valid_mask]
        return np.mean((final_residuals[valid_mask] - loo_preds[valid_mask])**2)

    best_power = golden_section_search(objective_function, 0, 20, tol=0.1)
    
    print("Computing out-of-sample similarities (Aggressive Multithreading)...")
    sim_matrix_all = np.zeros((N_all, len(src_idxs)), dtype=np.float32)
    batch_size = 20000
    
    from concurrent.futures import ThreadPoolExecutor
    
    def compute_batch(start_idx):
        end = min(start_idx + batch_size, N_all)
        s_tags = np.dot(f_tags[start_idx:end], f_tags[src_idxs].T)
        s_desc = np.dot(f_desc[start_idx:end], f_desc[src_idxs].T)
        s_verbs = np.dot(f_verbs[start_idx:end], f_verbs[src_idxs].T)
        s_graph = np.dot(f_graph[start_idx:end], f_graph[src_idxs].T) * pop_discount[start_idx:end, None]
        res = (weights['tags'] * s_tags + weights['desc'] * s_desc + weights['verbs'] * s_verbs + weights['graph'] * s_graph)
        return start_idx, end, res

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_batch, i) for i in range(0, N_all, batch_size)]
        for future in futures:
            start, end, res = future.result()
            sim_matrix_all[start:end] = res
        
    for j in range(len(src_idxs)):
        src_subg = subgenres_src[j]
        src_prob = subv_probs_src[j]
        if src_subg != 'Generic/Other':
            mask = (subgenres_all != 'Generic/Other') & (subgenres_all != src_subg)
            sim_matrix_all[mask, j] -= 0.3
        if src_prob > 0:
            joint_probs = np.sqrt(src_prob * subv_probs_all)
            sim_matrix_all[:, j] += (0.45 * joint_probs)

    weight_matrix = np.sign(sim_matrix_all) * (np.abs(sim_matrix_all) ** best_power)
    for j, idx in enumerate(src_idxs): weight_matrix[idx, j] = 0.0
        
    sum_abs_w = np.sum(np.abs(weight_matrix), axis=1)
    target_residual_pred = np.zeros(N_all)
    valid_mask = sum_abs_w > 0
    target_residual_pred[valid_mask] = np.sum(weight_matrix[valid_mask] * final_residuals, axis=1) / sum_abs_w[valid_mask]
    
    final_preds = base_preds_all + target_residual_pred
    final_preds = np.clip(final_preds, 0, 10)
    df['projected_rating'] = final_preds

    total_variance = np.mean((actual_ratings - np.mean(actual_ratings))**2)
    final_mse = objective_function(best_power)
    oos_r2 = 1.0 - (final_mse / total_variance)

    # --- FILTERS ---
    p = df['positive'].fillna(0).values
    n = df['negative'].fillna(0).values
    total_reviews = p + n
    pos_ratio = np.divide(p, total_reviews + 1e-9)
    
    base_mask = (
        (total_reviews >= 63) & 
        (pos_ratio >= 0.65) & 
        (~df['is_hollow'].fillna(False)) & 
        (~df['is_delisted'].fillna(False)) & 
        (~df['is_utility'].fillna(False)) & 
        (~df['is_nsfw'].fillna(False))
    )
    
    now = pd.Timestamp.now()
    parsed_dt = pd.to_datetime(df['parsed_date'], errors='coerce')
    is_future_exact = parsed_dt > now
    rel_date_str = df['release_date'].astype(str).str.lower()
    is_tba = rel_date_str.str.contains('coming|tba|tbd|announced|soon', na=False)
    curr_yr = now.year
    future_yrs = '|'.join([str(curr_yr + i) for i in range(1, 6)])
    is_future_year = rel_date_str.str.contains(future_yrs, na=False)
    q_str = r'q[234]' if now.month <= 3 else r'q[34]' if now.month <= 6 else r'q4' if now.month <= 9 else r'q_none'
    is_q_future = rel_date_str.str.contains(f'{q_str}\\s*{curr_yr}', regex=True, na=False)
    is_just_current_year = rel_date_str == str(curr_yr)
    is_future_text = is_tba | is_future_year | is_q_future | is_just_current_year
    upcoming_mask = (is_future_exact | is_future_text) & (total_reviews < 50) & (~df['is_hollow'].fillna(False)) & (~df['is_nsfw'].fillna(False))
    
    discovery_mask = base_mask & (~df['appid'].isin(all_gt_appids))
    is_free = df['tags'].apply(lambda x: 'Free to Play' in get_list(x))
    
    # 1. Top Recs
    top_discovery = df[discovery_mask].sort_values('projected_rating', ascending=False).head(30)
    top_recommendations = top_discovery[['appid', 'name', 'header_image', 'is_nsfw', 'projected_rating']].copy()
    
    # 2. Free Recs
    top_free = df[discovery_mask & is_free].sort_values('projected_rating', ascending=False).head(30)
    free_recommendations = top_free[['appid', 'name', 'header_image', 'is_nsfw', 'projected_rating']].copy()
    
    # 3. Backlog Recs
    backlog_recs = df[df['appid'].isin(backlog_appids)].sort_values('projected_rating', ascending=False).head(30)
    backlog_recommendations = backlog_recs[['appid', 'name', 'header_image', 'is_nsfw', 'projected_rating']].copy()
    
    # 4. Upcoming Recs
    upcoming_recs = df[upcoming_mask & (~df['appid'].isin(all_gt_appids))].sort_values('projected_rating', ascending=False).head(30)
    upcoming_recommendations = upcoming_recs[['appid', 'name', 'header_image', 'is_nsfw', 'projected_rating']].copy()

    # 5. North Stars (Highest Kernel Residual Pull)
    ns_scores = target_residual_pred.copy()
    ns_scores[~discovery_mask] = -1e12
    top_ns_idx = np.argsort(ns_scores)[-5:][::-1]
    north_stars = df.iloc[top_ns_idx][['appid', 'name', 'header_image', 'is_nsfw']].copy()
    north_stars['alignment'] = ns_scores[top_ns_idx]

    # 6. Interactive Pool (All valid games + features) for snappy frontend slider
    # First, gather ALL valid unowned + backlog games, excluding ignored, rated, and played
    exclude_statuses = ['ignored', 'rated', 'played']
    excluded_interactive_appids = set(gt[gt['status'].isin(exclude_statuses)]['appid'].tolist())
    
    interactive_mask = base_mask & (~df['appid'].isin(excluded_interactive_appids))
    top_interactive = df[interactive_mask].copy()
    
    interactive_pool = []
    # Intercept is beta[0], quality is beta[1], diff is beta[2] etc. based on selected_features
    # Create a mapping of beta coefficients
    coef_map = {}
    for i, feature in enumerate(selected_feature_names):
        coef_map[feature.lower()] = float(final_beta[i+1])
    
    for idx in top_interactive.index:
        game = df.iloc[idx]
        features = {}
        for feature in selected_feature_names:
            features[feature.lower()] = float(global_features[feature][idx])
        
        interactive_pool.append({
            'appid': int(game['appid']),
            'name': str(game['name']),
            'header_image': str(game['header_image']),
            'projected_rating': float(game['projected_rating']),
            'features': features,
            'kernel_residual': float(target_residual_pred[idx])
        })

    # 7. Associative Tags
    tag_impacts_formatted = []
    for stat in tag_stats[:15]:
        tag_mask = discovery_mask & df['tags'].apply(lambda x: stat['tag'] in get_list(x))
        sub_df = df[tag_mask].sort_values('projected_rating', ascending=False).head(5)
        tag_top_games = sub_df[['appid', 'name', 'header_image', 'is_nsfw']].to_dict(orient='records')
        tag_impacts_formatted.append({
            'tag': stat['tag'],
            'impact': float(stat['diff']),
            'top_games': tag_top_games
        })
    associative_tags = {'top': tag_impacts_formatted, 'bottom': []}

    # 8. Favorite Neighbors
    favorite_recs = []
    favorites = gt_rated[gt_rated['actual_rating'] >= 9.0].sort_values('actual_rating', ascending=False)
    fav_merged = favorites.merge(df[['appid', 'name']], on='appid', how='inner')
    fav_merged['meta_idx'] = fav_merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    valid_indices = np.where(discovery_mask | df['appid'].isin(backlog_appids))[0]
    
    for _, fav in fav_merged.iterrows():
        f_idx = fav['meta_idx']
        sim_total = sim_matrix_all[valid_indices, np.where(src_idxs == f_idx)[0][0]]
        top_100_idx = np.argsort(sim_total)[-100:][::-1]
        top_100_valid = valid_indices[top_100_idx]
        top_100_projs = final_preds[top_100_valid]
        best_10_idx = np.argsort(top_100_projs)[-10:][::-1]
        best_10_valid = top_100_valid[best_10_idx]
        
        fav_neighbors = df.iloc[best_10_valid][['appid', 'name', 'header_image', 'is_nsfw']].copy()
        fav_neighbors['predicted_rating'] = final_preds[best_10_valid]
        
        name_val = fav['name_x'] if 'name_x' in fav and pd.notna(fav['name_x']) else fav.get('name_y', fav.get('name', 'Unknown'))
        
        favorite_recs.append({
            'seed_appid': int(fav['appid']),
            'seed_name': str(name_val),
            'seed_header': str(df.iloc[f_idx]['header_image']),
            'seed_is_nsfw': bool(df.iloc[f_idx]['is_nsfw']),
            'top_games': fav_neighbors.to_dict(orient='records')
        })

    # Prepare final JSON
    result = {
        'metadata': {
            'quality': float(coef_map.get('quality', 0.0)), 
            'age': float(coef_map.get('age', 0.0)), 
            'popularity': float(coef_map.get('popularity', 0.0)), 
            'length': float(coef_map.get('length', 0.0)), 
            'difficulty': float(coef_map.get('difficulty', 0.0)), 
            'price': float(coef_map.get('price', 0.0)),
            'tone': float(coef_map.get('tone', 0.0)),
            'kernel_match': 1.0, 
            'best_q_idx': int(best_row), 
            'oos_r2': float(oos_r2),
            'similarity_power': float(best_power)
        },
        'kernel_anchors': [],
        'r2': float(oos_r2),
        'intercept': float(final_beta[0]),
        'library_appids': list(all_gt_appids),
        'rated_appids': gt_rated['appid'].tolist(),
        'top_recommendations': top_recommendations.to_dict(orient='records'),
        'free_recommendations': free_recommendations.to_dict(orient='records'),
        'backlog_recommendations': backlog_recommendations.to_dict(orient='records'),
        'upcoming_recommendations': upcoming_recommendations.to_dict(orient='records'),
        'north_stars': north_stars.to_dict(orient='records'),
        'associative_tags': associative_tags,
        'favorite_game_recommendations': favorite_recs,
        'interactive_pool': interactive_pool # <--- The snappy payload!
    }

    if output_path:
        with open(output_path, 'w') as f: json.dump(result, f, indent=4)
        np.save(output_path.replace('_taste_profile.json', '_predicted_ratings.npy'), final_preds.astype(np.float32))
        print(f"\n>>> SUCCESS: RICH TASTE DNA SAVED TO {output_path} (R2: {oos_r2:.4f}) <<<")
    return result

if __name__ == "__main__":
    sid = sys.argv[1]
    solve_user_taste(f"data/user_{sid}_ground_truth.csv", output_path=f"data/user_{sid}_taste_profile.json")
