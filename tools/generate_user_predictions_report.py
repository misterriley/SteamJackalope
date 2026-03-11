import pandas as pd
import numpy as np
import scipy.stats
import math
import os
from sklearn.linear_model import Ridge
from collections import defaultdict
import multiprocessing

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
            if 'Dating Sim' in intersecting_innocent:
                prob = max(prob, 0.95)
            elif 'Cute' in intersecting_innocent or 'Family Friendly' in intersecting_innocent:
                prob = max(prob, 0.85)
            else:
                prob = max(prob, 0.60)
        if has_satire:
            if 'Farming Sim' in intersecting_innocent or 'Game Development' in intersecting_innocent:
                prob = max(prob, 0.40)
            else:
                prob = max(prob, 0.25)
        if has_surreal:
            if 'Education' in intersecting_innocent or 'Math' in intersecting_innocent:
                prob = max(prob, 0.30)
            else:
                prob = max(prob, 0.15)
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

def main():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    N_all = len(df)
    
    gt_path = 'data/user_76561198039155404_ground_truth.csv'
    gt = pd.read_csv(gt_path)
    gt_rated = gt[gt['status'] == 'rated'].copy()
    all_gt_appids = set(gt['appid'].tolist())
    
    unplayed_statuses = ['backlog', 'unplayed', 'wishlist']
    backlog_appids = set(gt[gt['status'].isin(unplayed_statuses)]['appid'].tolist())
    if len(backlog_appids) == 0:
        backlog_appids = set(gt[~gt['status'].isin(['rated', 'ignored', 'played'])]['appid'].tolist())
    
    merged = gt_rated.merge(df[['appid']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    src_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    
    # ---------------------------------------------------------
    # STEP 1: PREDICTIVE TAGS ANALYSIS
    # ---------------------------------------------------------
    print("Analyzing predictive tags...")
    tag_lists_src = [set(get_list(t)) for t in df.iloc[src_idxs]['tags']]
    
    all_tags = set()
    for t_set in tag_lists_src:
        all_tags.update(t_set)
        
    tag_stats = []
    for tag in all_tags:
        with_tag = []
        without_tag = []
        for i, t_set in enumerate(tag_lists_src):
            if tag in t_set:
                with_tag.append(actual_ratings[i])
            else:
                without_tag.append(actual_ratings[i])
                
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
    top_predictive_tags = [t['tag'] for t in tag_stats[:5]]
    
    # ---------------------------------------------------------
    # STEP 2: LOAD FEATURES
    # ---------------------------------------------------------
    print("Loading feature matrices into RAM for faster parallel processing...")
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy'))
    f_desc = normalize(np.load('data/production/embeddings_desc.npy'))
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy').astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy'))
    
    pop_z = df['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    # ---------------------------------------------------------
    # STEP 3: DYNAMIC BASELINE SELECTION (QUALITY GRID + AICC)
    # ---------------------------------------------------------
    print("Finding Best Quality Grid Level...")
    quality_grid = np.load('data/production/quality_scores_grid.npy', mmap_mode='r')
    
    best_corr = -1
    best_row = -1
    for i in range(quality_grid.shape[0]):
        q = quality_grid[i, src_idxs]
        corr, _ = scipy.stats.pearsonr(q, actual_ratings)
        if corr > best_corr:
            best_corr = corr
            best_row = i
            
    print(f"-> Selected Quality Row: {best_row}")
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
        best_candidate = None
        best_candidate_aicc = current_aicc
        
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
            
    print(f"-> Final Selected Baseline Features: {selected_feature_names}")
    _, _, final_residuals, final_beta = calc_aicc(actual_ratings, X_train_current)
    
    X_global_selected = np.column_stack([global_features[f] for f in selected_feature_names])
    X_global_mat = np.column_stack((np.ones(N_all), X_global_selected))
    base_preds_all = X_global_mat @ final_beta
    
    # ---------------------------------------------------------
    # STEP 4: TUNING SIMILARITY KERNEL
    # ---------------------------------------------------------
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
        mse = np.mean((final_residuals[valid_mask] - loo_preds[valid_mask])**2)
        return mse

    best_power = golden_section_search(objective_function, 0, 20, tol=0.1)
    print(f"-> Optimal Similarity Power: {best_power:.2f}")

    # ---------------------------------------------------------
    # STEP 5: OUT-OF-SAMPLE PREDICTIONS (AGGRESSIVE BATCHING)
    # ---------------------------------------------------------
    print("Computing out-of-sample similarities (Aggressive Batching via BLAS threads)...")
    sim_matrix_all = np.zeros((N_all, len(src_idxs)), dtype=np.float32)
    batch_size = 20000
    for i in range(0, N_all, batch_size):
        end = min(i + batch_size, N_all)
        s_tags = np.dot(f_tags[i:end], f_tags[src_idxs].T)
        s_desc = np.dot(f_desc[i:end], f_desc[src_idxs].T)
        s_verbs = np.dot(f_verbs[i:end], f_verbs[src_idxs].T)
        s_graph = np.dot(f_graph[i:end], f_graph[src_idxs].T) * pop_discount[i:end, None]
        
        sim_matrix_all[i:end] = (
            weights['tags'] * s_tags +
            weights['desc'] * s_desc +
            weights['verbs'] * s_verbs +
            weights['graph'] * s_graph
        )
        
    print("Applying puzzle and subversion modifiers...")
    for j in range(len(src_idxs)):
        src_subg = subgenres_src[j]
        src_prob = subv_probs_src[j]
        
        if src_subg != 'Generic/Other':
            mask = (subgenres_all != 'Generic/Other') & (subgenres_all != src_subg)
            sim_matrix_all[mask, j] -= 0.3
            
        if src_prob > 0:
            joint_probs = np.sqrt(src_prob * subv_probs_all)
            sim_matrix_all[:, j] += (0.45 * joint_probs)

    print(f"Computing residual smoothing using optimal power {best_power:.2f}...")
    weight_matrix = np.sign(sim_matrix_all) * (np.abs(sim_matrix_all) ** best_power)
    
    # Exclude self-similarity
    for j, idx in enumerate(src_idxs):
        weight_matrix[idx, j] = 0.0
        
    sum_abs_w = np.sum(np.abs(weight_matrix), axis=1)
    target_residual_pred = np.zeros(N_all)
    valid_mask = sum_abs_w > 0
    target_residual_pred[valid_mask] = np.sum(weight_matrix[valid_mask] * final_residuals, axis=1) / sum_abs_w[valid_mask]
    
    final_preds = base_preds_all + target_residual_pred
    final_preds = np.clip(final_preds, 0, 10)
    df['projected_rating'] = final_preds

    # ---------------------------------------------------------
    # STEP 6: GENERATE REPORT
    # ---------------------------------------------------------
    p = df['positive'].fillna(0).values
    n = df['negative'].fillna(0).values
    total_reviews = p + n
    pos_ratio = np.divide(p, total_reviews + 1e-9)
    
    valid_game_mask = (
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
    
    q_str = ''
    if now.month <= 3: q_str = r'q[234]'
    elif now.month <= 6: q_str = r'q[34]'
    elif now.month <= 9: q_str = r'q4'
    else: q_str = r'q_none'
    
    is_q_future = rel_date_str.str.contains(f'{q_str}\\s*{curr_yr}', regex=True, na=False)
    is_just_current_year = rel_date_str == str(curr_yr)
    
    is_future_text = is_tba | is_future_year | is_q_future | is_just_current_year
    
    upcoming_mask = (
        (is_future_exact | is_future_text) & 
        (total_reviews < 50) & 
        (~df['is_hollow'].fillna(False)) & 
        (~df['is_nsfw'].fillna(False))
    )
    
    print("Generating report sections...")
    lines = ["# 🎮 User Game Predictions Report\n"]
    lines.append(f"**Regression details:** Quality Grid Row {best_row} | Baseline Features: {selected_feature_names} | Similarity Power: {best_power:.2f}\n")
    lines.append("This report contains personalized game recommendations generated by the Jackalope Kernel. It includes predictions for various categories, neighborhood insights based on your favorite games, and an analysis of tags that significantly boost your enjoyment.\n")
    
    def add_list(title, mask, n=30, sort_desc=True):
        lines.append(f"## {title}")
        lines.append("--------------------------------------------------")
        sub_df = df[mask].sort_values('projected_rating', ascending=(not sort_desc))
        for i, (_, row) in enumerate(sub_df.head(n).iterrows()):
            lines.append(f"{i+1:2d}. **{str(row['name'])}** - Proj: {row['projected_rating']:.2f}")
        lines.append("\n")

    add_list("Top 30 Recommended Games (Unowned)", valid_game_mask & (~df['appid'].isin(all_gt_appids)))
    add_list("Top 30 Priority Backlog Games", df['appid'].isin(backlog_appids))

    # A game is truly free if its price string says "Free" or it has a very low price_z
    is_free = (df['price_z'] < -1.0) | (df['price'].fillna('').str.lower().str.contains('free', na=False))
    add_list("Top 30 Free To Play Games", valid_game_mask & (~df['appid'].isin(all_gt_appids)) & is_free)
    add_list("Top 30 Highly Anticipated Upcoming Games", upcoming_mask & (~df['appid'].isin(all_gt_appids)))
    
    lines.append("## 📈 Highly Predictive Tags")
    lines.append("The following tags appear in your library and have passed a two-sided t-test (p < 0.05). Games with these tags score significantly higher in your personal ratings than games without them.\n")
    
    for stat in tag_stats[:10]:
        lines.append(f"### Tag: `{stat['tag']}`")
        lines.append(f"- **Avg Rating With Tag:** {stat['mean_with']:.2f} (n={stat['count']})")
        lines.append(f"- **Avg Rating Without:** {stat['mean_without']:.2f}")
        lines.append(f"- **P-Value:** {stat['p_val']:.4f}\n")
        
        tag_mask = valid_game_mask & (~df['appid'].isin(all_gt_appids - backlog_appids)) & df['tags'].apply(lambda x: stat['tag'] in get_list(x))
        sub_df = df[tag_mask].sort_values('projected_rating', ascending=False)
        lines.append(f"**Top 10 Games with `{stat['tag']}`:**")
        for i, (_, row) in enumerate(sub_df.head(10).iterrows()):
            lines.append(f"  {i+1}. {str(row['name'])} (Proj: {row['projected_rating']:.2f})")
        lines.append("\n")

    lines.append("## 🏘️ Neighborhood Recommendations for Your Favorites")
    lines.append("For each game you rated 9.0 or higher, here are the top 10 best games selected from its 100 closest structural neighbors.\n")
    
    favorites = gt_rated[gt_rated['actual_rating'] >= 9.0].sort_values('actual_rating', ascending=False)
    fav_merged = favorites.merge(df[['appid', 'name']], on='appid', how='inner')
    fav_merged['meta_idx'] = fav_merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    valid_indices = np.where(valid_game_mask & (~df['appid'].isin(all_gt_appids - backlog_appids)))[0]
    
    for _, fav in fav_merged.iterrows():
        idx = fav['meta_idx']
        name_col = 'name' if 'name' in fav else 'name_y'
        name = fav[name_col]
        rating = fav['actual_rating']
        
        # Pull similarity array for just this favorite against all valid unowned games
        sim_total = sim_matrix_all[valid_indices, np.where(src_idxs == idx)[0][0]]
        
        top_100_idx = np.argsort(sim_total)[-100:][::-1]
        top_100_valid = valid_indices[top_100_idx]
        
        top_100_projs = final_preds[top_100_valid]
        best_10_idx = np.argsort(top_100_projs)[-10:][::-1]
        best_10_valid = top_100_valid[best_10_idx]
        
        lines.append(f"### ★ {name} (You rated: {rating:.1f})")
        for rank, v_idx in enumerate(best_10_valid):
            v_name = df.iloc[v_idx]['name']
            v_proj = final_preds[v_idx]
            lines.append(f"{rank+1}. **{v_name}** - Proj: {v_proj:.2f}")
        lines.append("\n")

    out_file = 'user_game_predictions.md'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    print(f"Done! Full report written to {out_file}")

if __name__ == "__main__":
    main()
