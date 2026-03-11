import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

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
    
    gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
    gt_rated = gt[gt['status'] == 'rated'].copy()
    all_gt_appids = set(gt['appid'].tolist())
    
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
    q_vector = quality_grid[20, :]
    
    X_global_all = np.column_stack((q_vector, date_z))
    X_train = X_global_all[src_idxs]
    y_train = actual_ratings
    
    # Train Baseline Model
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
    subgenres_all = [identify_puzzle_subgenre(t) for t in tags_list_all]
    subv_scores_all = np.array([calculate_subversion_score(t) for t in tags_list_all])
    
    subgenres_src = [subgenres_all[idx] for idx in src_idxs]
    subv_scores_src = [subv_scores_all[idx] for idx in src_idxs]
    
    for j in range(len(src_idxs)):
        src_subg = subgenres_src[j]
        src_subv = subv_scores_src[j]
        
        if src_subg != 'Generic/Other':
            mask = np.array([(tg != 'Generic/Other' and tg != src_subg) for tg in subgenres_all])
            sim_matrix[mask, j] -= 0.3
            
        if src_subv >= 3.0:
            mask_3 = (subv_scores_all >= 3.0)
            mask_2 = (subv_scores_all == 2.0)
            mask_less = (subv_scores_all < 2.0)
            sim_matrix[mask_3, j] += 0.45
            sim_matrix[mask_2, j] += 0.25
            sim_matrix[mask_less, j] -= 0.30
        elif src_subv >= 2.0:
            mask_2 = (subv_scores_all >= 2.0)
            mask_less = (subv_scores_all < 2.0)
            sim_matrix[mask_2, j] += 0.25
            sim_matrix[mask_less, j] -= 0.20
        elif src_subv == 0.0:
            mask_2 = (subv_scores_all >= 2.0)
            sim_matrix[mask_2, j] -= 0.30

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
    
    unrated_mask = ~df['appid'].isin(all_gt_appids)
    
    df_valid = df[valid_game_mask & unrated_mask].copy()
    df_valid = df_valid.sort_values(by='projected_rating', ascending=False)
    
    print("\n--- Top 20 Projected Masterpieces ---")
    for i, (_, row) in enumerate(df_valid.head(20).iterrows()):
        print(f"{i+1:2d}. {str(row['name'])[:45]:<45} | Proj Rating: {row['projected_rating']:.2f}")

    print("\n--- Bottom 10 Projected Misses ---")
    for i, (_, row) in enumerate(df_valid.tail(10).iterrows()):
        print(f"{len(df_valid)-9+i}. {str(row['name'])[:45]:<45} | Proj Rating: {row['projected_rating']:.2f}")
        
    np.save('data/user_76561198039155404_projected_ratings_research.npy', final_preds.astype(np.float32))
    print("\nSaved full projected ratings to data/user_76561198039155404_projected_ratings_research.npy")

if __name__ == "__main__":
    main()
