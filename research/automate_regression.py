import pandas as pd
import numpy as np
import scipy.stats
import math
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
            if 'Dating Sim' in intersecting_innocent: prob = max(prob, 0.95)
            elif 'Cute' in intersecting_innocent or 'Family Friendly' in intersecting_innocent: prob = max(prob, 0.85)
            else: prob = max(prob, 0.60)
        if has_satire:
            if 'Farming Sim' in intersecting_innocent or 'Game Development' in intersecting_innocent: prob = max(prob, 0.40)
            else: prob = max(prob, 0.25)
        if has_surreal:
            if 'Education' in intersecting_innocent or 'Math' in intersecting_innocent: prob = max(prob, 0.30)
            else: prob = max(prob, 0.15)
    return prob

def calc_aicc(y, X):
    X_mat = np.column_stack((np.ones(len(y)), X))
    beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    preds = X_mat @ beta
    resid = y - preds
    rss = np.sum(resid**2)
    n = len(y)
    k = X_mat.shape[1]
    aic = n * np.log(rss / n) + 2 * k
    if n - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = np.inf
    return aicc, preds, resid

def golden_section_search(f, a, b, tol=0.1):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = f(c)
    fd = f(d)
    
    print(f"Starting GSS in [{a}, {b}]")
    step = 1
    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - resphi * (b - a)
            fd = f(d)
        print(f"Step {step}: Interval is now [{a:.3f}, {b:.3f}]")
        step += 1
            
    return (b + a) / 2

def main():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    
    gt_path = 'data/user_76561198039155404_ground_truth.csv'
    gt = pd.read_csv(gt_path)
    gt_rated = gt[gt['status'] == 'rated'].copy()
    
    merged = gt_rated.merge(df[['appid']], on='appid', how='inner')
    merged['meta_idx'] = merged['appid'].map({appid: idx for idx, appid in enumerate(df['appid'])})
    src_idxs = merged['meta_idx'].values
    actual_ratings = merged['actual_rating'].values
    
    print("\n--- STEP 1: Find Best Quality Grid Level ---")
    quality_grid = np.load('data/production/quality_scores_grid.npy', mmap_mode='r')
    
    best_corr = -1
    best_row = -1
    for i in range(quality_grid.shape[0]):
        q = quality_grid[i, src_idxs]
        corr, _ = scipy.stats.pearsonr(q, actual_ratings)
        if corr > best_corr:
            best_corr = corr
            best_row = i
            
    print(f"Best Quality Row: {best_row} with Pearson Correlation {best_corr:.4f}")
    quality_feature = quality_grid[best_row, src_idxs]
    
    print("\n--- STEP 2: Forward Selection of Linear Predictors (AICc) ---")
    features = {
        'Quality': quality_feature,
        'Popularity': df['pop_z'].fillna(0).values[src_idxs],
        'Price': df['price_z'].fillna(0).values[src_idxs],
        'Age': df['date_z'].fillna(0).values[src_idxs],
        'Length': df['playtime_z'].fillna(0).values[src_idxs],
        'Difficulty': df['difficulty_z'].fillna(0).values[src_idxs],
        'Tone': df['tone_z'].fillna(0).values[src_idxs]
    }
    
    selected_features = ['Quality']
    remaining_features = ['Popularity', 'Price', 'Age', 'Length', 'Difficulty', 'Tone']
    
    X_current = features['Quality'].reshape(-1, 1)
    current_aicc, base_preds, current_residuals = calc_aicc(actual_ratings, X_current)
    print(f"Base AICc (Quality only): {current_aicc:.2f}")
    
    while remaining_features:
        best_candidate = None
        best_candidate_aicc = current_aicc
        
        for candidate in remaining_features:
            X_test = np.column_stack((X_current, features[candidate]))
            test_aicc, _, _ = calc_aicc(actual_ratings, X_test)
            if test_aicc < best_candidate_aicc:
                best_candidate_aicc = test_aicc
                best_candidate = candidate
                
        if best_candidate:
            print(f"+ Added '{best_candidate}' | New AICc: {best_candidate_aicc:.2f}")
            selected_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            X_current = np.column_stack((X_current, features[best_candidate]))
            current_aicc = best_candidate_aicc
        else:
            print("No more features improved AICc.")
            break
            
    print(f"\nFinal Selected Baseline Features: {selected_features}")
    
    # Recalculate residuals from final linear model
    _, _, final_residuals = calc_aicc(actual_ratings, X_current)
    
    print("\n--- STEP 3: Tuning Similarity Power (Golden Section Search) ---")
    print("Loading feature matrices for similarity...")
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))
    
    pop_z = df['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    sim_tags = np.dot(f_tags[src_idxs], f_tags[src_idxs].T)
    sim_desc = np.dot(f_desc[src_idxs], f_desc[src_idxs].T)
    sim_verbs = np.dot(f_verbs[src_idxs], f_verbs[src_idxs].T)
    sim_graph = np.dot(f_graph[src_idxs], f_graph[src_idxs].T) * pop_discount[src_idxs][:, None]
    
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}
    sim_matrix = (
        weights['tags'] * sim_tags +
        weights['desc'] * sim_desc +
        weights['verbs'] * sim_verbs +
        weights['graph'] * sim_graph
    )
    
    tags_list_all = [set(get_list(x)) for x in df['tags']]
    subgenres_all = np.array([identify_puzzle_subgenre(t) for t in tags_list_all])
    subv_probs_all = np.array([calculate_subversion_probability(t) for t in tags_list_all])
    
    subgenres_src = subgenres_all[src_idxs]
    subv_probs_src = subv_probs_all[src_idxs]
    
    for j in range(len(src_idxs)):
        src_subg = subgenres_src[j]
        src_prob = subv_probs_src[j]
        
        if src_subg != 'Generic/Other':
            mask = (subgenres_src != 'Generic/Other') & (subgenres_src != src_subg)
            sim_matrix[mask, j] -= 0.3
            
        if src_prob > 0:
            joint_probs = np.sqrt(src_prob * subv_probs_src)
            sim_matrix[:, j] += (0.45 * joint_probs)
            
    # Set self-similarity to 0 for LOO
    np.fill_diagonal(sim_matrix, 0)
    
    def objective_function(power):
        weight_matrix = np.sign(sim_matrix) * (np.abs(sim_matrix) ** power)
        sum_abs_w = np.sum(np.abs(weight_matrix), axis=1)
        
        valid_mask = sum_abs_w > 0
        loo_preds = np.zeros(len(final_residuals))
        if np.any(valid_mask):
            loo_preds[valid_mask] = np.sum(weight_matrix[valid_mask] * final_residuals, axis=1) / sum_abs_w[valid_mask]
        
        mse = np.mean((final_residuals[valid_mask] - loo_preds[valid_mask])**2)
        return mse

    best_power = golden_section_search(objective_function, 0, 20, tol=0.1)
    print(f"\nOptimal Similarity Power (Golden Section Search): {best_power:.2f}")
    
    best_mse = objective_function(best_power)
    
    # Calculate Final Out-Of-Sample R^2 over all predictions vs mean
    total_variance = np.mean((actual_ratings - np.mean(actual_ratings))**2)
    # The sum of MSE from linear + LOO kernel residual preds
    # Wait, the objective_function just returns MSE of the residual prediction.
    # Total LOO preds:
    best_weight_matrix = np.sign(sim_matrix) * (np.abs(sim_matrix) ** best_power)
    sum_abs_w = np.sum(np.abs(best_weight_matrix), axis=1)
    valid_mask = sum_abs_w > 0
    final_loo_preds = np.zeros(len(actual_ratings))
    
    # Base preds
    _, linear_preds, _ = calc_aicc(actual_ratings, X_current)
    
    if np.any(valid_mask):
        final_loo_preds[valid_mask] = linear_preds[valid_mask] + (np.sum(best_weight_matrix[valid_mask] * final_residuals, axis=1) / sum_abs_w[valid_mask])
        
    final_mse = np.mean((actual_ratings[valid_mask] - final_loo_preds[valid_mask])**2)
    final_r2 = 1.0 - (final_mse / total_variance)
    
    print(f"Final Combined R^2 (Out-of-sample similarity): {final_r2:.4f}")

if __name__ == "__main__":
    main()
