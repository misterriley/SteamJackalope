import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

def evaluate_signed_kernel(sim_matrix, actual_ratings, power, ridge=0.0):
    N = len(actual_ratings)
    # The new transform: sign(sim) * |sim|^p
    weight_matrix = np.sign(sim_matrix) * (np.abs(sim_matrix) ** power)
    np.fill_diagonal(weight_matrix, 0)
    
    predictions = []
    global_mean = np.mean(actual_ratings)
    
    for i in range(N):
        w = weight_matrix[i]
        sum_abs_w = np.sum(np.abs(w))
        
        # Add ridge parameter to the denominator to pull low-confidence predictions toward the mean
        pred = global_mean + np.sum(w * (actual_ratings - global_mean)) / (sum_abs_w + ridge)
            
        # Clip to valid 0-10 bounds just in case
        pred = np.clip(pred, 0, 10)
        predictions.append(pred)
        
    predictions = np.array(predictions)
    r2 = r2_score(actual_ratings, predictions)
    mae = mean_absolute_error(actual_ratings, predictions)
    return r2, mae

def main():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    
    gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
    gt = gt[gt['status'] == 'rated'].copy()
    
    merged = gt.merge(df[['appid']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    valid_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    N = len(valid_idxs)
    
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r')[valid_idxs])
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r')[valid_idxs])
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r')[valid_idxs].astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r')[valid_idxs])
    
    pop_z = df.iloc[valid_idxs]['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    sim_tags = np.dot(f_tags, f_tags.T)
    sim_desc = np.dot(f_desc, f_desc.T)
    sim_verbs = np.dot(f_verbs, f_verbs.T)
    sim_graph = np.dot(f_graph, f_graph.T) * pop_discount[None, :]
    
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}
    
    sim_matrix = (
        weights['tags'] * sim_tags +
        weights['desc'] * sim_desc +
        weights['verbs'] * sim_verbs +
        weights['graph'] * sim_graph
    )
    
    tags_list = [set(get_list(x)) for x in df.iloc[valid_idxs]['tags']]
    subgenres = [identify_puzzle_subgenre(t) for t in tags_list]
    subv_scores = [calculate_subversion_score(t) for t in tags_list]
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if subgenres[i] != 'Generic/Other':
                if subgenres[j] != 'Generic/Other' and subgenres[j] != subgenres[i]:
                    sim_matrix[i, j] -= 0.3
            t_subv = subv_scores[i]
            m_subv = subv_scores[j]
            if t_subv >= 3.0:
                if m_subv >= 3.0: sim_matrix[i, j] += 0.45
                elif m_subv >= 2.0: sim_matrix[i, j] += 0.25
                else: sim_matrix[i, j] -= 0.30
            elif t_subv >= 2.0:
                if m_subv >= 2.0: sim_matrix[i, j] += 0.25
                else: sim_matrix[i, j] -= 0.20
            elif t_subv == 0.0:
                if m_subv >= 2.0: sim_matrix[i, j] -= 0.30
                
    results = []
    
    print("\n--- Grid Search: Signed Polynomial + Ridge Regularization ---")
    powers_to_test = [6, 8, 10, 12, 14, 16]
    ridges_to_test = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    for p in powers_to_test:
        for r in ridges_to_test:
            r2, mae = evaluate_signed_kernel(sim_matrix.copy(), actual_ratings, power=p, ridge=r)
            results.append((p, r, r2, mae))

    results.sort(key=lambda x: x[2], reverse=True)
    
    print("Top 10 Configurations (by R2):")
    for res in results[:10]:
        print(f"Power: {res[0]:<4} | Ridge: {res[1]:<5} | R2: {res[2]:.4f} | MAE: {res[3]:.4f}")

    best = results[0]
    print(f"\nBEST OVERALL: Power {best[0]}, Ridge {best[1]} (R2: {best[2]:.4f}, MAE: {best[3]:.4f})")

if __name__ == "__main__":
    main()
