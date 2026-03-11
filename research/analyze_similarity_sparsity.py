import pandas as pd
import numpy as np
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
    
    gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
    gt_rated = gt[gt['status'] == 'rated'].copy()
    all_gt_appids = set(gt['appid'].tolist())
    
    # Get user's favorites (9s and 10s)
    favorites = gt_rated[gt_rated['actual_rating'] >= 9.0].copy()
    favorites = favorites.sort_values(by='actual_rating', ascending=False)
    
    merged = favorites.merge(df[['appid', 'name']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    fav_idxs = merged['meta_idx'].values
    name_col = 'name' if 'name' in merged.columns else 'name_y'
    fav_names = merged[name_col].values
    
    print("Loading feature matrices...")
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))
    
    pop_z = df['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    tags_list_all = [set(get_list(x)) for x in df['tags']]
    subgenres_all = np.array([identify_puzzle_subgenre(t) for t in tags_list_all])
    subv_scores_all = np.array([calculate_subversion_score(t) for t in tags_list_all])
    
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
        (~df['is_nsfw'].fillna(False)) &
        (~df['appid'].isin(all_gt_appids))
    )
    valid_indices = np.where(valid_game_mask)[0]
    
    print("\nAnalyzing Similarity Drop-offs for Top 100 Neighbors...")
    print(f"{'Game Name':<35} | {'1st':<6} | {'10th':<6} | {'50th':<6} | {'100th':<6} | {'Drop (1st-100th)'}")
    print("-" * 80)
    
    results = []
    
    for i in range(len(fav_idxs)):
        idx = fav_idxs[i]
        name = fav_names[i]
        
        sim_tags = np.dot(f_tags[valid_indices], f_tags[idx])
        sim_desc = np.dot(f_desc[valid_indices], f_desc[idx])
        sim_verbs = np.dot(f_verbs[valid_indices], f_verbs[idx])
        sim_graph = np.dot(f_graph[valid_indices], f_graph[idx]) * pop_discount[valid_indices]
        
        sim_total = (0.174 * sim_tags) + (0.445 * sim_desc) + (0.233 * sim_verbs) + (0.148 * sim_graph)
        
        # Apply modifiers
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
            
        # Top 100 closest sorted descending
        top_100_sims = np.sort(sim_total)[-100:][::-1]
        
        sim_1 = top_100_sims[0]
        sim_10 = top_100_sims[9]
        sim_50 = top_100_sims[49]
        sim_100 = top_100_sims[99]
        drop = sim_1 - sim_100
        
        results.append({
            'name': name,
            'sim_1': sim_1,
            'sim_10': sim_10,
            'sim_50': sim_50,
            'sim_100': sim_100,
            'drop': drop
        })
        
    # Sort by drop to see the sparest neighborhoods
    results.sort(key=lambda x: x['drop'], reverse=True)
    
    for r in results:
        print(f"{str(r['name'])[:35]:<35} | {r['sim_1']:.3f} | {r['sim_10']:.3f} | {r['sim_50']:.3f} | {r['sim_100']:.3f} | {r['drop']:.3f}")

if __name__ == "__main__":
    main()
