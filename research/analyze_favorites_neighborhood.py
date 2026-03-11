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
    proj_ratings = np.load('data/user_76561198039155404_projected_ratings_research.npy')
    df['projected_rating'] = proj_ratings
    
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
    fav_ratings = merged['actual_rating'].values
    
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
    
    print("Processing 100 Closest Neighbors for each Favorite Game...")
    
    output_lines = ["# Your Personalized Neighborhood Recommendations\n"]
    output_lines.append("For each game you rated 9 or 10, we found its 100 closest structural neighbors (based on Tags, Verbs, Descriptions, and Graph). Then, we ranked those 100 games by their overall Projected Rating to give you the absolute highest quality matches.\n")
    
    for i in range(len(fav_idxs)):
        idx = fav_idxs[i]
        name = fav_names[i]
        rating = fav_ratings[i]
        
        sim_tags = np.dot(f_tags[valid_indices], f_tags[idx])
        sim_desc = np.dot(f_desc[valid_indices], f_desc[idx])
        sim_verbs = np.dot(f_verbs[valid_indices], f_verbs[idx])
        
        # Discount the graph similarity by the popularity of the neighbor being considered
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
            
        # Top 100 closest
        top_100_idx = np.argsort(sim_total)[-100:][::-1]
        top_100_valid = valid_indices[top_100_idx]
        
        # Sort these 100 by projected rating
        top_100_projs = proj_ratings[top_100_valid]
        best_10_idx = np.argsort(top_100_projs)[-10:][::-1]
        best_10_valid = top_100_valid[best_10_idx]
        
        output_lines.append(f"\n### ★ {name} (You rated: {rating:.1f})")
        
        for rank, v_idx in enumerate(best_10_valid):
            v_name = df.iloc[v_idx]['name']
            v_proj = proj_ratings[v_idx]
            v_sim = sim_total[top_100_idx[best_10_idx[rank]]]
            output_lines.append(f"{rank+1}. **{v_name}** - Proj: {v_proj:.2f} (Sim Score: {v_sim:.3f})")
            
    with open('neighborhood_recommendations.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print("Done! Saved full report to 'neighborhood_recommendations.md'")

if __name__ == "__main__":
    main()
