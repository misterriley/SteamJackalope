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

def main():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    
    # Load user data
    gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
    gt = gt[gt['status'] == 'rated'].copy()
    
    # Merge to get metadata indices
    merged = gt.merge(df[['appid']], on='appid', how='inner')
    # get the integer indices in the original metadata dataframe
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    valid_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    names = merged['name'].values
    N = len(valid_idxs)
    
    print(f"Calculating similarity matrix for {N} rated games...")
    
    # Load features only for the valid indices to save memory
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r')[valid_idxs])
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r')[valid_idxs])
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r')[valid_idxs].astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r')[valid_idxs])
    
    pop_z = df.iloc[valid_idxs]['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    # Base similarities
    sim_tags = np.dot(f_tags, f_tags.T)
    sim_desc = np.dot(f_desc, f_desc.T)
    sim_verbs = np.dot(f_verbs, f_verbs.T)
    sim_graph = np.dot(f_graph, f_graph.T) * pop_discount[None, :] # broadcast discount over columns
    
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}
    
    sim_matrix = (
        weights['tags'] * sim_tags +
        weights['desc'] * sim_desc +
        weights['verbs'] * sim_verbs +
        weights['graph'] * sim_graph
    )
    
    # Apply Puzzle and Subversion modifiers
    tags_list = [set(get_list(x)) for x in df.iloc[valid_idxs]['tags']]
    subgenres = [identify_puzzle_subgenre(t) for t in tags_list]
    subv_scores = [calculate_subversion_score(t) for t in tags_list]
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            
            # Puzzle Firewall
            if subgenres[i] != 'Generic/Other':
                if subgenres[j] != 'Generic/Other' and subgenres[j] != subgenres[i]:
                    sim_matrix[i, j] -= 0.3
                    
            # Subversion
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
    
    # We will use an exponential kernel or a power kernel to transform similarities into weights.
    # Power kernel: max(0, sim)^power.
    power = 8.0
    weight_matrix = np.maximum(0, sim_matrix) ** power
    
    # Zero out the diagonal (hold-out)
    np.fill_diagonal(weight_matrix, 0)
    
    predictions = []
    
    for i in range(N):
        w = weight_matrix[i]
        sum_w = np.sum(w)
        if sum_w > 0:
            pred = np.sum(w * actual_ratings) / sum_w
        else:
            pred = np.mean(actual_ratings)
        predictions.append(pred)
        
    predictions = np.array(predictions)
    
    r2 = r2_score(actual_ratings, predictions)
    mae = mean_absolute_error(actual_ratings, predictions)
    rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
    
    print("\n--- Kernel Regression Results ---")
    print(f"Total Rated Games Evaluated: {N}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Let's show the top 10 best predictions and top 10 worst predictions
    errors = np.abs(predictions - actual_ratings)
    sorted_idx = np.argsort(errors)
    
    print("\n--- Top 10 Most Accurate Predictions ---")
    for idx in sorted_idx[:10]:
        print(f"Game: {str(names[idx])[:40]:<40} | Actual: {actual_ratings[idx]:.1f} | Predicted: {predictions[idx]:.2f} | Error: {errors[idx]:.2f}")
        
    print("\n--- Top 10 Least Accurate Predictions ---")
    for idx in sorted_idx[-10:]:
        print(f"Game: {str(names[idx])[:40]:<40} | Actual: {actual_ratings[idx]:.1f} | Predicted: {predictions[idx]:.2f} | Error: {errors[idx]:.2f}")

if __name__ == "__main__":
    main()
