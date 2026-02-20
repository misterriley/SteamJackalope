import pandas as pd
import numpy as np
import os
import sys
import json
import ast
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE,
    ROOT_DIR,
    TAG_PRIOR_COUNTS_FILE,
    TAG_PRIOR_TRANSFORMED_FILE,
    TAG_TRANSFORM_TYPE,
    TAG_GLOBAL_SCALING_FACTOR,
    DOT_PRODUCT_LAMBDA
)

def load_unwhitened_vectors(csv_path):
    """
    Reconstructs the unwhitened transformed tag vectors.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Load Prior G
    print("Loading tag priors...")
    G_final = np.load(TAG_PRIOR_COUNTS_FILE)
    
    # Load Constants
    with open(os.path.join(ROOT_DIR, "data", "production", "regularization_constants.json"), "r") as f:
        constants = json.load(f)
    K = constants["TAG_VECTOR_K"]
    
    # Parse Tags
    print("Parsing tags...")
    all_game_tags = []
    for tag_str in tqdm(df['tags'], desc="Scanning", smoothing=0):
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
    num_tags = len(unique_tags)
    num_games = len(all_game_tags)
    
    counts = np.zeros((num_games, num_tags), dtype=np.float32)
    for i, tags in enumerate(all_game_tags):
        for t, c in tags.items():
            if t in tag_to_idx:
                counts[i, tag_to_idx[t]] = c
                
    total_votes = counts.sum(axis=1)
    
    print(f"Applying Bayesian regularization (K={K:.2f}) and transform ({TAG_TRANSFORM_TYPE})...")
    N = total_votes.reshape(-1, 1)
    profiles = (counts + K * G_final) / (N + K)
    
    if TAG_TRANSFORM_TYPE == 'clr':
        log_v = np.log(profiles + 1e-9)
        gm_log = log_v.mean(axis=1, keepdims=True)
        V = log_v - gm_log
    elif TAG_TRANSFORM_TYPE == 'anscombe':
        avg_n = np.mean(total_votes)
        V = 2 * np.sqrt(profiles * avg_n + 0.375)
        V = V / V.sum(axis=1, keepdims=True)
    else:
        V = profiles
        
    return V, df['appid'].values

def run_tuning_experiment(ground_truth_path, unwhitened_vectors, appids, sample_size=None):
    print(f"\n--- Tuning Whitening Dimensionality (Sample Size: {sample_size or 'Full'}) ---")
    df_gt = pd.read_csv(ground_truth_path)
    df_gt = df_gt.dropna(subset=['actual_rating'])
    if 'ignore' in df_gt.columns:
        df_gt = df_gt[df_gt['ignore'] == False]
        
    if sample_size and len(df_gt) > sample_size:
        df_gt = df_gt.sample(n=sample_size, random_state=42)
        print(f"Subsampled to {sample_size} ratings.")

    user_appids = df_gt['appid'].values
    y = df_gt['actual_rating'].values
    
    appid_to_idx = {aid: i for i, aid in enumerate(appids)}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    y = y[[aid in appid_to_idx for aid in user_appids]]
    
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    q_user = quality_grid[10][user_indices]
    
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z'])
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z']
    meta_user = full_metadata.iloc[user_indices][meta_cols].values
    
    print("Performing Global SVD on unwhitened tag space...")
    n_samples = unwhitened_vectors.shape[0]
    M = np.dot(unwhitened_vectors.T, unwhitened_vectors) / n_samples
    U, S, Vt = np.linalg.svd(M)
    cumvar = np.cumsum(S) / np.sum(S)
    
    results = []
    k_range = [10, 20, 30, 40, 50, 75, 100, 150, 200, 243, 300, 350, 389, 450]
    
    print(f"{'K':>4} | {'Var %':>7} | {'R^2':>7} | {'MSE':>7} | {'Alpha':>8}")
    print("-" * 55)
    
    for k in k_range:
        if k > len(S): break
        
        U_k = U[:, :k]
        S_k = S[:k]
        W_k = np.dot(U_k, np.diag(1.0 / np.sqrt(S_k + 1e-6)))
        
        user_tags_whitened = np.dot(unwhitened_vectors[user_indices], W_k)
        norms = np.linalg.norm(user_tags_whitened, axis=1, keepdims=True)
        user_tags_scaled = (user_tags_whitened / (norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
        
        X = np.hstack([q_user.reshape(-1, 1), meta_user, user_tags_scaled])
        
        alphas = np.logspace(-2, 6, 41)
        model = RidgeCV(alphas=alphas)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        var_pct = cumvar[k-1] * 100
        print(f"{k:4d} | {var_pct:6.2f}% | {r2:7.4f} | {mse:7.4f} | {model.alpha_:8.1f}")
        
    return results

if __name__ == "__main__":
    gt_path = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    csv_path = "data/pipeline_games_clean.csv"
    
    unwhitened, appids = load_unwhitened_vectors(csv_path)
    run_tuning_experiment(gt_path, unwhitened, appids, sample_size=sample_size)
