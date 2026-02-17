import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import sys

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

def run_meta_study(steamid):
    gt_path = f"data/user_{steamid}_ground_truth.csv"
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found: {gt_path}")
        return

    print(f"Loading ground truth from {gt_path}...")
    df_full = pd.read_csv(gt_path)
    df_all = df_full[df_full['ignore'] == False].copy()
    df_all = df_all.dropna(subset=['actual_rating'])
    
    total_samples = len(df_all)
    sample_sizes = list(range(50, total_samples, 50))
    if total_samples not in sample_sizes:
        sample_sizes.append(total_samples)
        
    print(f"Total available rated games: {total_samples}")
    print(f"Will test sample sizes: {sample_sizes}")

    print(f"Loading metadata and tag vectors...")
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'pop_z', 'date_z', 'playtime_z', 'difficulty_z'])
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    
    # Load Tag Vectors and Norms (full set)
    tag_vectors_full = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms_full = np.load(TAG_NORMS_FILE, mmap_mode='r')
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    
    meta_results = []
    
    # Pre-calculate metadata for all relevant games
    all_user_indices = [appid_to_idx[aid] for aid in df_all['appid'].values if aid in appid_to_idx]
    
    for n in sample_sizes:
        print(f"\n--- Processing Sample Size N = {n} ---")
        # Subsample
        df_sub = df_all.sample(n=n, random_state=42)
        user_appids = df_sub['appid'].values
        y = df_sub['actual_rating'].values
        user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
        
        # Optimal Discovery for this subsample
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
        
        # Prepare features
        tag_vectors_raw = tag_vectors_full[user_indices]
        full_norms = tag_norms_full[user_indices].reshape(-1, 1)
        meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z']
        user_meta_features = np.clip(full_metadata.iloc[user_indices][meta_cols].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)

        # ADAPTIVE ITERATIONS: More for low N, fewer for high N
        current_max_iter = 20000 if n < 150 else 5000
        
        # Prepare for K comparison: K=1 vs K=N-6
        max_possible_k = tag_vectors_raw.shape[1]
        k_n = np.clip(n - 6, 1, max_possible_k)
        k_values = [1, k_n]
        
        n_results = {}
        
        for k in k_values:
            user_tag_features = tag_vectors_raw[:, :k].astype(np.float32)
            user_tag_features_norm = user_tag_features / (full_norms + DOT_PRODUCT_LAMBDA)
            user_tag_features_scaled = user_tag_features_norm * TAG_GLOBAL_SCALING_FACTOR
            
            X = np.hstack([q_global.reshape(-1, 1), user_meta_features, user_tag_features_scaled])
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            
            dummy_X = np.zeros((1, X_scaled.shape[1]))
            dummy_y = np.array([5.0])
            X_with_dummy = np.vstack([X_scaled, dummy_X])
            y_with_dummy = np.append(y, dummy_y)
            
            model = LassoCV(cv=5, max_iter=current_max_iter, selection='random', tol=1e-3)
            model.fit(X_with_dummy, y_with_dummy)
            
            mean_mse = np.mean(model.mse_path_, axis=1)
            best_alpha_idx = np.argmin(mean_mse)
            ss_res_cv = mean_mse[best_alpha_idx] * len(y_with_dummy)
            ss_tot = np.sum((y_with_dummy - np.mean(y_with_dummy))**2)
            r2_cv = 1 - (ss_res_cv / ss_tot)
            
            n_results[k] = r2_cv
            
        r2_k1 = n_results[1]
        r2_kn = n_results[k_n]
        delta_r2 = r2_k1 - r2_kn
        
        print(f"  Result: K=1 R^2: {r2_k1:.4f}, K={k_n} R^2: {r2_kn:.4f}, Delta: {delta_r2:.4f}")
        
        meta_results.append({
            'n': n,
            'k_n': k_n,
            'r2_k1': r2_k1,
            'r2_kn': r2_kn,
            'delta_r2': delta_r2
        })

    # Final Plot: Delta R^2 vs N (K=1 vs K=N-6)
    meta_df = pd.DataFrame(meta_results)
    plt.figure(figsize=(10, 6))
    plt.plot(meta_df['n'], meta_df['delta_r2'], marker='o', linestyle='-', color='tab:red', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Sample Size (N)')
    plt.ylabel('R^2 Loss (K=1 - K=N-6)')
    plt.title(f'Saturated DNA Test: K=1 vs K=N-6 Predictive Loss\nUser: {steamid}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f"research/dna_complexity_trend_{steamid}.png")
    print(f"\nTrend plot saved to research/dna_complexity_trend_{steamid}.png")
    
    # Also save the raw data for analysis
    meta_df.to_json(f"research/meta_study_results_{steamid}.json", indent=4)
    plt.show()

if __name__ == "__main__":
    steamid = "76561198039155404"
    run_meta_study(steamid)
