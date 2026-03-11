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

def evaluate_kernel(sim_matrix, actual_ratings, transform_func, **kwargs):
    N = len(actual_ratings)
    weight_matrix = transform_func(sim_matrix, **kwargs)
    np.fill_diagonal(weight_matrix, 0)
    
    predictions = []
    global_mean = np.mean(actual_ratings)
    
    for i in range(N):
        w = weight_matrix[i]
        sum_w = np.sum(w)
        if sum_w > 0:
            pred = np.sum(w * actual_ratings) / sum_w
        else:
            pred = global_mean
        predictions.append(pred)
        
    predictions = np.array(predictions)
    r2 = r2_score(actual_ratings, predictions)
    mae = mean_absolute_error(actual_ratings, predictions)
    return r2, mae

# Transforms
def poly_transform(sim, power):
    return np.maximum(0, sim) ** power

def exp_transform(sim, beta):
    # Shift by max for numerical stability before exp
    shifted_sim = sim - np.max(sim, axis=1, keepdims=True)
    return np.exp(beta * shifted_sim)

def thresh_poly_transform(sim, power, threshold):
    return np.maximum(0, sim - threshold) ** power

def top_k_transform(sim, k):
    W = np.zeros_like(sim)
    for i in range(len(sim)):
        # get indices of top k
        idx = np.argsort(sim[i])[-k:]
        W[i, idx] = sim[i, idx]
        # set non-positive sims to 0 even if in top k
        W[i] = np.maximum(0, W[i])
    return W

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
    
    print("\n--- Polynomial Transform (max(0, sim)^p) ---")
    for p in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]:
        r2, mae = evaluate_kernel(sim_matrix.copy(), actual_ratings, poly_transform, power=p)
        print(f"Power: {p:<3} | R2: {r2:.4f} | MAE: {mae:.4f}")
        results.append(('Poly', p, r2, mae))

    print("\n--- Exponential Transform (exp(beta * sim)) ---")
    for b in [1, 3, 5, 10, 15, 20, 30, 50]:
        r2, mae = evaluate_kernel(sim_matrix.copy(), actual_ratings, exp_transform, beta=b)
        print(f"Beta:  {b:<3} | R2: {r2:.4f} | MAE: {mae:.4f}")
        results.append(('Exp', b, r2, mae))

    print("\n--- Thresholded Polynomial (max(0, sim - thresh)^p) ---")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for p in [1, 3, 5, 10]:
            r2, mae = evaluate_kernel(sim_matrix.copy(), actual_ratings, thresh_poly_transform, power=p, threshold=t)
            print(f"Thresh: {t}, Power: {p:<2} | R2: {r2:.4f} | MAE: {mae:.4f}")
            results.append((f'ThreshPoly(t={t})', p, r2, mae))
            
    print("\n--- Top K Neighbors ---")
    for k in [3, 5, 10, 20, 50]:
        r2, mae = evaluate_kernel(sim_matrix.copy(), actual_ratings, top_k_transform, k=k)
        print(f"Top K: {k:<3} | R2: {r2:.4f} | MAE: {mae:.4f}")
        results.append(('TopK', k, r2, mae))

    # Best by R2
    best = max(results, key=lambda x: x[2])
    print(f"\nBEST TRANSFORMATION: {best[0]} with param={best[1]} (R2: {best[2]:.4f}, MAE: {best[3]:.4f})")

if __name__ == "__main__":
    main()
