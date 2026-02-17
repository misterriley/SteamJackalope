import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)

def run_parametric_study_subsample(steamid, sample_size=100):
    gt_path = f"data/user_{steamid}_ground_truth.csv"
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found: {gt_path}")
        return

    print(f"Loading ground truth from {gt_path}...")
    df_full = pd.read_csv(gt_path)
    df = df_full[df_full['ignore'] == False].copy()
    df = df.dropna(subset=['actual_rating'])
    
    # Subsample 100 games
    if len(df) > sample_size:
        print(f"Subsampling {sample_size} games from {len(df)} rated games...")
        df = df.sample(n=sample_size, random_state=42)
    else:
        print(f"Library size {len(df)} is already <= {sample_size}.")

    user_appids = df['appid'].values
    y = df['actual_rating'].values
    
    print(f"Loading metadata and tag vectors...")
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z'])
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    
    if len(user_indices) != len(user_appids):
        found_mask = [aid in appid_to_idx for aid in user_appids]
        y = y[found_mask]
        user_indices = [idx for idx in user_indices if idx is not None]

    # Find optimal Discovery
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    num_steps = quality_grid.shape[0]
    correlations = []
    for i in range(num_steps):
        q_step = quality_grid[i][user_indices]
        if np.std(q_step) > 1e-9 and np.std(y) > 1e-9:
            corr = np.corrcoef(q_step, y)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)
    
    best_idx = np.argmax(np.abs(correlations))
    q_global = np.clip(quality_grid[best_idx][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)

    # Load Tag Vectors and Norms
    tag_vectors_raw = np.load(TAG_VECTORS_FILE, mmap_mode='r')[user_indices]
    full_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')[user_indices].reshape(-1, 1)
    
    # Load Metadata Features
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z']
    user_meta_features = full_metadata.iloc[user_indices][meta_cols].values
    user_meta_features = np.clip(user_meta_features, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)

    # Prepare for loop
    max_k = tag_vectors_raw.shape[1]
    results = []
    alphas = np.logspace(-2, 6, 81)
    
    print(f"Starting parametric sweep (N=100) from K=1 to {max_k} (Full Granularity)...")
    for k in range(1, max_k + 1):
        user_tag_features = tag_vectors_raw[:, :k].astype(np.float32)
        user_tag_features_norm = user_tag_features / (full_norms + DOT_PRODUCT_LAMBDA)
        user_tag_features_scaled = user_tag_features_norm * TAG_GLOBAL_SCALING_FACTOR
        
        X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled])
        
        # Add dummy game
        dummy_X = np.zeros((1, X.shape[1]))
        dummy_y = np.array([5.0])
        X_with_dummy = np.vstack([X, dummy_X])
        y_with_dummy = np.append(y, dummy_y)
        
        from sklearn.linear_model import LassoCV
        # Increase max_iter for convergence on high-dim tag space
        model = LassoCV(cv=5, max_iter=10000)
        model.fit(X_with_dummy, y_with_dummy)
        
        # 1. Training R^2 (In-sample fit)
        r2_train = model.score(X_with_dummy, y_with_dummy)
        
        # 2. Cross-Validated R^2 (Predictive performance)
        # For LassoCV, we use the mse_path_ to find the best CV score
        # mse_path_ is (n_alphas, n_folds)
        mean_mse = np.mean(model.mse_path_, axis=1)
        best_alpha_idx = np.argmin(mean_mse)
        ss_res_cv = mean_mse[best_alpha_idx] * len(y_with_dummy)
        ss_tot = np.sum((y_with_dummy - np.mean(y_with_dummy))**2)
        r2_cv = 1 - (ss_res_cv / ss_tot)
        
        alpha = model.alpha_
        
        results.append({
            'k': k, 
            'r2_train': r2_train, 
            'r2_cv': r2_cv, 
            'alpha': alpha
        })
        
        if k % 20 == 0 or k == max_k:
            print(f"  K={k:3d}: Train R^2={r2_train:.4f}, CV R^2={r2_cv:.4f}, Alpha={alpha:.4f}")

    results_df = pd.DataFrame(results)
    suffix = f"subsample_{sample_size}"
    results_df.to_csv(f"research/dna_parametric_results_{steamid}_{suffix}.csv", index=False)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Dimensions (K)')
    ax1.set_ylabel('R^2 Score')
    
    ax1.plot(results_df['k'], results_df['r2_train'], color='tab:green', label='Training R^2 (In-Sample)')
    ax1.plot(results_df['k'], results_df['r2_cv'], color='tab:blue', label='LOOCV R^2 (Predictive)')
    
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Optimal Alpha (log scale)', color=color)
    ax2.plot(results_df['k'], results_df['alpha'], color=color, label='Optimal Alpha', linestyle='--')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'DNA Generalization Gap (N={sample_size}): Train vs CV R^2\nUser: {steamid}')
    fig.tight_layout()
    plt.savefig(f"research/dna_parametric_plot_{steamid}_{suffix}.png")
    print(f"Plot saved to research/dna_parametric_plot_{steamid}_{suffix}.png")

if __name__ == "__main__":
    steamid = "76561198039155404"
    run_parametric_study_subsample(steamid, 50)
