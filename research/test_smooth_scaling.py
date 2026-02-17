import pandas as pd
import numpy as np
import os
import sys
import json
import ast
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE,
    ROOT_DIR,
    TAG_PRIOR_COUNTS_FILE,
    TAG_GLOBAL_SCALING_FACTOR,
    DOT_PRODUCT_LAMBDA,
    TAG_TRANSFORM_TYPE
)

def load_unwhitened_vectors(csv_path):
    df = pd.read_csv(csv_path)
    G_final = np.load(TAG_PRIOR_COUNTS_FILE)
    with open(os.path.join(ROOT_DIR, "data", "production", "regularization_constants.json"), "r") as f:
        constants = json.load(f)
    K = constants["TAG_VECTOR_K"]
    
    all_game_tags = []
    for tag_str in df['tags']:
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
            all_game_tags.append({})
            continue
        try:
            tags_dict = ast.literal_eval(tag_str)
            all_game_tags.append(tags_dict if isinstance(tags_dict, dict) else {})
        except:
            all_game_tags.append({})
            
    unique_tags = sorted(list(set(tag for tags in all_game_tags for tag in tags.keys())))
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    counts = np.zeros((len(all_game_tags), len(unique_tags)), dtype=np.float32)
    for i, tags in enumerate(all_game_tags):
        for t, c in tags.items():
            if t in tag_to_idx:
                counts[i, tag_to_idx[t]] = c
                
    total_votes = counts.sum(axis=1)
    N = total_votes.reshape(-1, 1)
    profiles = (counts + K * G_final) / (N + K)
    
    if TAG_TRANSFORM_TYPE == 'clr':
        log_v = np.log(profiles + 1e-9)
        V = log_v - log_v.mean(axis=1, keepdims=True)
    else:
        V = profiles
        
    return V, df['appid'].values

def test_linear_scaling(gt_path, unwhitened_vectors, appids):
    df_gt = pd.read_csv(gt_path)
    df_gt = df_gt.dropna(subset=['actual_rating'])
    if 'ignore' in df_gt.columns:
        df_gt = df_gt[df_gt['ignore'] == False]
    
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z'])
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    
    # Global SVD
    n_samples = unwhitened_vectors.shape[0]
    M = np.dot(unwhitened_vectors.T, unwhitened_vectors) / n_samples
    U, S, Vt = np.linalg.svd(M)
    
    sample_sizes = [50, 100, 150, 200, 250, len(df_gt)]
    max_k_available = 243 # The current production whitening
    
    print(f"{'N Ratings':>10} | {'Smooth K':>10} | {'R^2':>8} | {'Alpha':>10}")
    print("-" * 50)
    
    for n in sample_sizes:
        # Subsample
        df_sub = df_gt.sample(n=min(n, len(df_gt)), random_state=42)
        n_actual = len(df_sub)
        
        # Linear Formula
        k_smooth = int(np.clip(40 + 0.7 * n_actual, 40, max_k_available))
        
        # Prep Data
        user_appids = df_sub['appid'].values
        y = df_sub['actual_rating'].values
        appid_to_idx = {aid: i for i, aid in enumerate(appids)}
        user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
        y = y[[aid in appid_to_idx for aid in user_appids]]
        
        q_user = quality_grid[10][user_indices]
        meta_user = full_metadata.iloc[user_indices][['date_z', 'pop_z', 'playtime_z', 'difficulty_z']].values
        
        # Whiten to k_smooth
        U_k = U[:, :k_smooth]
        S_k = S[:k_smooth]
        W_k = np.dot(U_k, np.diag(1.0 / np.sqrt(S_k + 1e-6)))
        
        user_tags_whitened = np.dot(unwhitened_vectors[user_indices], W_k)
        norms = np.linalg.norm(user_tags_whitened, axis=1, keepdims=True)
        user_tags_scaled = (user_tags_whitened / (norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
        
        X = np.hstack([q_user.reshape(-1, 1), meta_user, user_tags_scaled])
        
        # Ridge CV
        alphas = np.logspace(-2, 6, 41)
        model = RidgeCV(alphas=alphas)
        model.fit(X, y)
        
        print(f"{n_actual:10d} | {k_smooth:10d} | {model.score(X, y):8.4f} | {model.alpha_:10.1f}")

if __name__ == "__main__":
    gt_path = "data/user_76561198039155404_ground_truth.csv"
    unwhitened, appids = load_unwhitened_vectors("data/pipeline_games_clean.csv")
    test_linear_scaling(gt_path, unwhitened, appids)
