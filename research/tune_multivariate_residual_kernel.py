import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

def evaluate_multivariate_residual_kernel(sim_matrix, X_global, actual_ratings, power=10.0):
    N = len(actual_ratings)
    # The base signed polynomial kernel
    weight_matrix = np.sign(sim_matrix) * (np.abs(sim_matrix) ** power)
    np.fill_diagonal(weight_matrix, 0)
    
    predictions = []
    
    # We'll also track coefficients to see what matters globally
    all_coefs = []
    
    # 1. Base Prediction: Multivariate Linear Regression
    for i in range(N):
        # Hold out the target game
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        
        # Train linear model on the N-1 games
        X_train = X_global[mask]
        y_train = actual_ratings[mask]
        
        # We'll use Ridge to add a tiny bit of stability to the multivariate regression
        lr = Ridge(alpha=1.0)
        lr.fit(X_train, y_train)
        all_coefs.append(lr.coef_)
        
        # Calculate training residuals
        train_preds = lr.predict(X_train)
        residuals = y_train - train_preds
        
        # 2. Predict the target game's base score
        target_X = X_global[i].reshape(1, -1)
        target_base_pred = lr.predict(target_X)[0]
        
        # 3. Kernel Smoothing over the Residuals
        w = weight_matrix[i, mask]
        sum_abs_w = np.sum(np.abs(w))
        
        if sum_abs_w > 0:
            target_residual_pred = np.sum(w * residuals) / sum_abs_w
        else:
            target_residual_pred = 0.0 # If no neighbors, predict exactly the base score
            
        # 4. Final Prediction = Base + Residual
        final_pred = target_base_pred + target_residual_pred
        final_pred = np.clip(final_pred, 0, 10)
        predictions.append(final_pred)
        
    predictions = np.array(predictions)
    r2 = r2_score(actual_ratings, predictions)
    mae = mean_absolute_error(actual_ratings, predictions)
    
    avg_coefs = np.mean(all_coefs, axis=0)
    return r2, mae, predictions, avg_coefs

def main():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    
    gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
    gt = gt[gt['status'] == 'rated'].copy()
    
    merged = gt.merge(df[['appid']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    
    valid_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    names = merged['name'].values
    N = len(valid_idxs)
    
    # Load features
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r')[valid_idxs])
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r')[valid_idxs])
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r')[valid_idxs].astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r')[valid_idxs])
    
    pop_z = df.iloc[valid_idxs]['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    # Load global signals
    quality_grid = np.load('data/production/quality_scores_grid.npy', mmap_mode='r')
    q_vector = quality_grid[20, valid_idxs] 
    
    # Fill nan values for other signals with 0 (mean)
    date_z = df.iloc[valid_idxs]['date_z'].fillna(0).values
    price_z = df.iloc[valid_idxs]['price_z'].fillna(0).values
    diff_z = df.iloc[valid_idxs]['difficulty_z'].fillna(0).values
    
    # Base similarities
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
    
    # Apply Puzzle and Subversion modifiers
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

    print("\nEvaluating Multivariate Residual Models...")
    
    # Test 1: Quality Only (Our previous best)
    X_q_only = q_vector.reshape(-1, 1)
    r2_q, mae_q, preds_q, coefs_q = evaluate_multivariate_residual_kernel(sim_matrix.copy(), X_q_only, actual_ratings)
    print(f"\n[1] Quality Only Base -> R2: {r2_q:.4f} | MAE: {mae_q:.4f}")
    print(f"    Avg Coefs: Quality={coefs_q[0]:.3f}")
    
    # Test 2: Quality + Age
    X_qa = np.column_stack((q_vector, date_z))
    r2_qa, mae_qa, preds_qa, coefs_qa = evaluate_multivariate_residual_kernel(sim_matrix.copy(), X_qa, actual_ratings)
    print(f"\n[2] Quality + Age Base -> R2: {r2_qa:.4f} | MAE: {mae_qa:.4f}")
    print(f"    Avg Coefs: Quality={coefs_qa[0]:.3f}, Date_Z={coefs_qa[1]:.3f}")

    # Test 3: Quality + Age + Price
    X_qap = np.column_stack((q_vector, date_z, price_z))
    r2_qap, mae_qap, preds_qap, coefs_qap = evaluate_multivariate_residual_kernel(sim_matrix.copy(), X_qap, actual_ratings)
    print(f"\n[3] Quality + Age + Price Base -> R2: {r2_qap:.4f} | MAE: {mae_qap:.4f}")
    print(f"    Avg Coefs: Quality={coefs_qap[0]:.3f}, Date_Z={coefs_qap[1]:.3f}, Price_Z={coefs_qap[2]:.3f}")
    
    # Test 4: Quality + Age + Price + Difficulty
    X_all = np.column_stack((q_vector, date_z, price_z, diff_z))
    r2_all, mae_all, preds_all, coefs_all = evaluate_multivariate_residual_kernel(sim_matrix.copy(), X_all, actual_ratings)
    print(f"\n[4] All Global Signals Base -> R2: {r2_all:.4f} | MAE: {mae_all:.4f}")
    print(f"    Avg Coefs: Quality={coefs_all[0]:.3f}, Date_Z={coefs_all[1]:.3f}, Price_Z={coefs_all[2]:.3f}, Diff_Z={coefs_all[3]:.3f}")

if __name__ == "__main__":
    main()
