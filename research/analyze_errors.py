import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from collections import Counter

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
    df = pd.read_parquet('data/production/metadata.parquet')
    
    gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
    gt = gt[gt['status'] == 'rated'].copy()
    
    merged = gt.merge(df[['appid', 'name']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    valid_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    name_col = 'name' if 'name' in merged.columns else 'name_y'
    names = merged[name_col].values
    N = len(valid_idxs)
    
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r')[valid_idxs])
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r')[valid_idxs])
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r')[valid_idxs].astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r')[valid_idxs])
    
    pop_z = df.iloc[valid_idxs]['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    quality_grid = np.load('data/production/quality_scores_grid.npy', mmap_mode='r')
    q_vector = quality_grid[20, valid_idxs] 
    date_z = df.iloc[valid_idxs]['date_z'].fillna(0).values
    
    sim_tags = np.dot(f_tags, f_tags.T)
    sim_desc = np.dot(f_desc, f_desc.T)
    sim_verbs = np.dot(f_verbs, f_verbs.T)
    sim_graph = np.dot(f_graph, f_graph.T) * pop_discount[None, :]
    
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}
    sim_matrix = weights['tags'] * sim_tags + weights['desc'] * sim_desc + weights['verbs'] * sim_verbs + weights['graph'] * sim_graph
    
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

    weight_matrix = np.sign(sim_matrix) * (np.abs(sim_matrix) ** 10.0)
    np.fill_diagonal(weight_matrix, 0)
    
    predictions = []
    X_global = np.column_stack((q_vector, date_z))
    
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        
        X_train = X_global[mask]
        y_train = actual_ratings[mask]
        
        lr = Ridge(alpha=1.0)
        lr.fit(X_train, y_train)
        
        train_preds = lr.predict(X_train)
        residuals = y_train - train_preds
        
        target_X = X_global[i].reshape(1, -1)
        target_base_pred = lr.predict(target_X)[0]
        
        w = weight_matrix[i, mask]
        sum_abs_w = np.sum(np.abs(w))
        
        if sum_abs_w > 0:
            target_residual_pred = np.sum(w * residuals) / sum_abs_w
        else:
            target_residual_pred = 0.0
            
        final_pred = target_base_pred + target_residual_pred
        final_pred = np.clip(final_pred, 0, 10)
        predictions.append(final_pred)
        
    predictions = np.array(predictions)
    errors = predictions - actual_ratings  # positive = overpredicted, negative = underpredicted
    
    # Analyze the errors
    results = []
    for i in range(N):
        results.append({
            'name': str(names[i]),
            'actual': actual_ratings[i],
            'predicted': predictions[i],
            'error': errors[i],
            'abs_error': abs(errors[i]),
            'tags': list(tags_list[i])
        })
        
    res_df = pd.DataFrame(results)
    
    print("=== MODEL SEVERELY UNDER-PREDICTS THESE GAMES ===")
    print("(You rated them much HIGHER than the model expected based on similar games)")
    under_preds = res_df.sort_values('error').head(15)
    for _, row in under_preds.iterrows():
        print(f"{row['name'][:35]:<35} | Actual: {row['actual']:.1f} | Proj: {row['predicted']:.2f} | Diff: {row['error']:.2f}")

    print("\n=== MODEL SEVERELY OVER-PREDICTS THESE GAMES ===")
    print("(You rated them much LOWER than the model expected based on similar games)")
    over_preds = res_df.sort_values('error', ascending=False).head(15)
    for _, row in over_preds.iterrows():
        print(f"{row['name'][:35]:<35} | Actual: {row['actual']:.1f} | Proj: {row['predicted']:.2f} | Diff: {row['error']:.2f}")

    # Look for common tags in the top 30 under-predicted and top 30 over-predicted
    under_tags = []
    for t_list in res_df.sort_values('error').head(30)['tags']:
        under_tags.extend(t_list)
        
    over_tags = []
    for t_list in res_df.sort_values('error', ascending=False).head(30)['tags']:
        over_tags.extend(t_list)
        
    print("\n=== TAG PATTERNS IN UNDER-PREDICTED (Games you love more than expected) ===")
    under_counts = Counter(under_tags)
    for t, c in under_counts.most_common(10):
        print(f"{t}: {c}")

    print("\n=== TAG PATTERNS IN OVER-PREDICTED (Games you dislike more than expected) ===")
    over_counts = Counter(over_tags)
    for t, c in over_counts.most_common(10):
        print(f"{t}: {c}")

if __name__ == "__main__":
    main()
