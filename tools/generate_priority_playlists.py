import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os

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

def main():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    N_all = len(df)
    
    gt_path = 'data/user_76561198039155404_ground_truth.csv'
    if not os.path.exists(gt_path):
        print(f"Error: {gt_path} not found.")
        return
        
    gt = pd.read_csv(gt_path)
    gt_rated = gt[gt['status'] == 'rated'].copy()
    all_gt_appids = set(gt['appid'].tolist())
    
    # Backlog: Games the user owns/wishes for but hasn't rated/played/ignored
    unplayed_statuses = ['backlog', 'unplayed', 'wishlist']
    backlog_appids = set(gt[gt['status'].isin(unplayed_statuses)]['appid'].tolist())
    if len(backlog_appids) == 0:
        # Fallback if statuses are different
        backlog_appids = set(gt[~gt['status'].isin(['rated', 'ignored', 'played'])]['appid'].tolist())
    
    merged = gt_rated.merge(df[['appid']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    src_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    
    print("Loading full feature matrices...")
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))
    
    pop_z = df['pop_z'].fillna(0).values
    date_z = df['date_z'].fillna(0).values
    quality_grid = np.load('data/production/quality_scores_grid.npy', mmap_mode='r')
    q_vector = quality_grid[20, :] # Discovery 1.0 quality scores
    
    X_global_all = np.column_stack((q_vector, date_z))
    X_train = X_global_all[src_idxs]
    y_train = actual_ratings
    
    print("Training Baseline Model (Quality + Age)...")
    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    
    # Get residuals for rated games
    train_preds = lr.predict(X_train)
    residuals = y_train - train_preds
    
    # Predict base for all games
    base_preds_all = lr.predict(X_global_all)
    
    print("Computing similarities (this may take a moment)...")
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
    
    print("Applying puzzle and subversion modifiers...")
    tags_list_all = [set(get_list(x)) for x in df['tags']]
    subgenres_all = np.array([identify_puzzle_subgenre(t) for t in tags_list_all])
    subv_probs_all = np.array([calculate_subversion_probability(t) for t in tags_list_all])
    
    subgenres_src = subgenres_all[src_idxs]
    subv_probs_src = subv_probs_all[src_idxs]
    
    for j in range(len(src_idxs)):
        src_subg = subgenres_src[j]
        src_prob = subv_probs_src[j]
        
        if src_subg != 'Generic/Other':
            mask = (subgenres_all != 'Generic/Other') & (subgenres_all != src_subg)
            sim_matrix[mask, j] -= 0.3
            
        if src_prob > 0:
            joint_probs = np.sqrt(src_prob * subv_probs_all)
            sim_matrix[:, j] += (0.45 * joint_probs)

    print("Computing residual smoothing (power = 10.0)...")
    weight_matrix = np.sign(sim_matrix) * (np.abs(sim_matrix) ** 10.0)
    
    # Exclude self-voting for rated games
    for j, idx in enumerate(src_idxs):
        weight_matrix[idx, j] = 0.0
        
    sum_abs_w = np.sum(np.abs(weight_matrix), axis=1)
    
    target_residual_pred = np.zeros(N_all)
    valid_mask = sum_abs_w > 0
    target_residual_pred[valid_mask] = np.sum(weight_matrix[valid_mask] * residuals, axis=1) / sum_abs_w[valid_mask]
    
    final_preds = base_preds_all + target_residual_pred
    final_preds = np.clip(final_preds, 0, 10)
    
    df['projected_rating'] = final_preds
    
    # Global quality/shovelware filters for out-of-catalogue games
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
    
    print("\n" + "="*60)
    print("🎮 YOUR PERSONALIZED PRIORITY PLAYLISTS")
    print("="*60)
    
    # --- 1. Top 30 In-Catalogue (Backlog) ---
    df_backlog = df[df['appid'].isin(backlog_appids)].copy()
    df_backlog = df_backlog.sort_values(by='projected_rating', ascending=False)
    
    print("\n📌 TOP 30 BACKLOG GAMES TO PLAY NEXT")
    print("--------------------------------------------------")
    for i, (_, row) in enumerate(df_backlog.head(30).iterrows()):
        print(f"{i+1:2d}. {str(row['name'])[:45]:<45} | Proj Rating: {row['projected_rating']:.2f}")

    # --- 2. Top 30 Out-of-Catalogue ---
    # Must pass quality filter, must not be in ground truth at all
    df_out = df[valid_game_mask & (~df['appid'].isin(all_gt_appids))].copy()
    df_out = df_out.sort_values(by='projected_rating', ascending=False)
    
    print("\n🚀 TOP 30 OUT-OF-CATALOGUE RECOMMENDATIONS")
    print("--------------------------------------------------")
    for i, (_, row) in enumerate(df_out.head(30).iterrows()):
        print(f"{i+1:2d}. {str(row['name'])[:45]:<45} | Proj Rating: {row['projected_rating']:.2f}")
        
    # --- 3. Top 30 Free Games ---
    def is_free(tags):
        return 'Free to Play' in get_list(tags)
        
    df_out['is_free'] = df_out['tags'].apply(is_free)
    df_free = df_out[df_out['is_free']].copy()
    
    print("\n💸 TOP 30 FREE GAMES")
    print("--------------------------------------------------")
    for i, (_, row) in enumerate(df_free.head(30).iterrows()):
        print(f"{i+1:2d}. {str(row['name'])[:45]:<45} | Proj Rating: {row['projected_rating']:.2f}")

if __name__ == "__main__":
    main()
