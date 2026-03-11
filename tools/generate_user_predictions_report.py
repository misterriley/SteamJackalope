import pandas as pd
import numpy as np
import os
import scipy.stats
from sklearn.linear_model import Ridge
from collections import defaultdict

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

def calculate_subversion_score(tags_set):
    meta_tags = {'Psychological Horror', 'Fourth Wall', 'Surreal', 'Satire', 'Parody', 'Illuminati', 'Mind-Bending'}
    innocent_tags = {'Cute', 'Education', 'Dating Sim', 'Family Friendly', 'Farming Sim', 'Typing', 'Math', 'Software', 'Game Development'}
    meta_count = len(tags_set.intersection(meta_tags))
    innocent_count = len(tags_set.intersection(innocent_tags))
    if meta_count >= 1 and innocent_count >= 1: return 3.0
    elif meta_count >= 2: return 2.0
    elif meta_count == 1: return 1.0
    return 0.0

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
    
    # Analyze predictive tags using t-test
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
            # Two-sided t-test
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
    print(f"Top predictive tags found: {top_predictive_tags}")
    
    print("Loading feature matrices...")
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))
    
    pop_z = df['pop_z'].fillna(0).values
    date_z = df['date_z'].fillna(0).values
    quality_grid = np.load('data/production/quality_scores_grid.npy', mmap_mode='r')
    q_vector = quality_grid[20, :]
    
    X_global_all = np.column_stack((q_vector, date_z))
    X_train = X_global_all[src_idxs]
    y_train = actual_ratings
    
    print("Training Baseline Model...")
    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    
    train_preds = lr.predict(X_train)
    residuals = y_train - train_preds
    base_preds_all = lr.predict(X_global_all)
    
    print("Computing similarities...")
    sim_tags = np.dot(f_tags, f_tags[src_idxs].T)
    sim_desc = np.dot(f_desc, f_desc[src_idxs].T)
    sim_verbs = np.dot(f_verbs, f_verbs[src_idxs].T)
    
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    sim_graph = np.dot(f_graph, f_graph[src_idxs].T) * pop_discount[:, None]
    
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}
    sim_matrix = (
        weights['tags'] * sim_tags +
        weights['desc'] * sim_desc +
        weights['verbs'] * sim_verbs +
        weights['graph'] * sim_graph
    )
    
    tags_list_all = [set(get_list(x)) for x in df['tags']]
    subgenres_all = np.array([identify_puzzle_subgenre(t) for t in tags_list_all])
    subv_scores_all = np.array([calculate_subversion_score(t) for t in tags_list_all])
    
    subgenres_src = subgenres_all[src_idxs]
    subv_scores_src = subv_scores_all[src_idxs]
    
    for j in range(len(src_idxs)):
        src_subg = subgenres_src[j]
        src_subv = subv_scores_src[j]
        if src_subg != 'Generic/Other':
            mask = (subgenres_all != 'Generic/Other') & (subgenres_all != src_subg)
            sim_matrix[mask, j] -= 0.3
        if src_subv >= 3.0:
            sim_matrix[subv_scores_all >= 3.0, j] += 0.45
            sim_matrix[subv_scores_all == 2.0, j] += 0.25
            sim_matrix[subv_scores_all < 2.0, j] -= 0.30
        elif src_subv >= 2.0:
            sim_matrix[subv_scores_all >= 2.0, j] += 0.25
            sim_matrix[subv_scores_all < 2.0, j] -= 0.20
        elif src_subv == 0.0:
            sim_matrix[subv_scores_all >= 2.0, j] -= 0.30

    print("Computing residual smoothing...")
    weight_matrix = np.sign(sim_matrix) * (np.abs(sim_matrix) ** 10.0)
    
    for j, idx in enumerate(src_idxs):
        weight_matrix[idx, j] = 0.0
        
    sum_abs_w = np.sum(np.abs(weight_matrix), axis=1)
    target_residual_pred = np.zeros(N_all)
    valid_mask = sum_abs_w > 0
    target_residual_pred[valid_mask] = np.sum(weight_matrix[valid_mask] * residuals, axis=1) / sum_abs_w[valid_mask]
    
    final_preds = base_preds_all + target_residual_pred
    final_preds = np.clip(final_preds, 0, 10)
    df['projected_rating'] = final_preds
    
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
    
    rel_date_str = df['release_date'].astype(str).str.lower()
    upcoming_mask = (
        rel_date_str.str.contains('coming|tba|tbd|announced|2025|2026|soon') & 
        (total_reviews < 50) & 
        (~df['is_hollow'].fillna(False)) & 
        (~df['is_nsfw'].fillna(False))
    )
    
    print("Generating report sections...")
    lines = ["# 🎮 User Game Predictions Report\n"]
    lines.append("This report contains personalized game recommendations generated by the Jackalope Kernel. It includes predictions for various categories, neighborhood insights based on your favorite games, and an analysis of tags that significantly boost your enjoyment.\n")
    
    def add_list(title, mask, n=30, sort_desc=True):
        lines.append(f"## {title}")
        lines.append("--------------------------------------------------")
        sub_df = df[mask].sort_values('projected_rating', ascending=(not sort_desc))
        for i, (_, row) in enumerate(sub_df.head(n).iterrows()):
            lines.append(f"{i+1:2d}. **{str(row['name'])}** - Proj: {row['projected_rating']:.2f}")
        lines.append("\n")

    # 1. Top 30 Unowned Games
    add_list("Top 30 Recommended Games (Unowned)", valid_game_mask & (~df['appid'].isin(all_gt_appids)))
    
    # 2. Top 30 Backlog
    add_list("Top 30 Priority Backlog Games", df['appid'].isin(backlog_appids))
    
    # 3. Top 30 Free Games
    is_free = df['tags'].apply(lambda x: 'Free to Play' in get_list(x))
    add_list("Top 30 Free To Play Games", valid_game_mask & (~df['appid'].isin(all_gt_appids)) & is_free)
    
    # 4. Top 30 Upcoming Games
    add_list("Top 30 Highly Anticipated Upcoming Games", upcoming_mask & (~df['appid'].isin(all_gt_appids)))
    
    # 5. Predictive Tags Breakdown
    lines.append("## 📈 Highly Predictive Tags")
    lines.append("The following tags appear in your library and have passed a two-sided t-test (p < 0.05). Games with these tags score significantly higher in your personal ratings than games without them.\n")
    
    for stat in tag_stats[:10]: # Top 10 predictive tags stats
        lines.append(f"### Tag: `{stat['tag']}`")
        lines.append(f"- **Avg Rating With Tag:** {stat['mean_with']:.2f} (n={stat['count']})")
        lines.append(f"- **Avg Rating Without:** {stat['mean_without']:.2f}")
        lines.append(f"- **P-Value:** {stat['p_val']:.4f}\n")
        
        tag_mask = valid_game_mask & (~df['appid'].isin(all_gt_appids)) & df['tags'].apply(lambda x: stat['tag'] in get_list(x))
        sub_df = df[tag_mask].sort_values('projected_rating', ascending=False)
        lines.append(f"**Top 10 Unowned Games with `{stat['tag']}`:**")
        for i, (_, row) in enumerate(sub_df.head(10).iterrows()):
            lines.append(f"  {i+1}. {str(row['name'])} (Proj: {row['projected_rating']:.2f})")
        lines.append("\n")

    # 6. Neighborhood Recommendations for Favorites
    lines.append("## 🏘️ Neighborhood Recommendations for Your Favorites")
    lines.append("For each game you rated 9.0 or higher, here are the top 10 best games selected from its 100 closest structural neighbors.\n")
    
    favorites = gt_rated[gt_rated['actual_rating'] >= 9.0].sort_values('actual_rating', ascending=False)
    fav_merged = favorites.merge(df[['appid', 'name']], on='appid', how='inner')
    fav_merged['meta_idx'] = fav_merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    valid_indices = np.where(valid_game_mask & (~df['appid'].isin(all_gt_appids)))[0]
    
    for _, fav in fav_merged.iterrows():
        idx = fav['meta_idx']
        name_col = 'name' if 'name' in fav else 'name_y'
        name = fav[name_col]
        rating = fav['actual_rating']
        
        sim_t = np.dot(f_tags[valid_indices], f_tags[idx])
        sim_d = np.dot(f_desc[valid_indices], f_desc[idx])
        sim_v = np.dot(f_verbs[valid_indices], f_verbs[idx])
        sim_g = np.dot(f_graph[valid_indices], f_graph[idx]) * pop_discount[valid_indices]
        
        sim_total = (0.174 * sim_t) + (0.445 * sim_d) + (0.233 * sim_v) + (0.148 * sim_g)
        
        src_subg = subgenres_all[idx]
        src_subv = subv_scores_all[idx]
        v_subgenres = subgenres_all[valid_indices]
        v_subv_scores = subv_scores_all[valid_indices]
        
        if src_subg != 'Generic/Other':
            mask = (v_subgenres != 'Generic/Other') & (v_subgenres != src_subg)
            sim_total[mask] -= 0.3
            
        if src_subv >= 3.0:
            sim_total[v_subv_scores >= 3.0] += 0.45
            sim_total[v_subv_scores == 2.0] += 0.25
            sim_total[v_subv_scores < 2.0] -= 0.30
        elif src_subv >= 2.0:
            sim_total[v_subv_scores >= 2.0] += 0.25
            sim_total[v_subv_scores < 2.0] -= 0.20
        elif src_subv == 0.0:
            sim_total[v_subv_scores >= 2.0] -= 0.30
            
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
