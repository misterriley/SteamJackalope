import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX

def research_difficulty_rbf(steamid="76561198039155404"):
    # 1. Load Data
    df_gt = pd.read_csv(f"data/user_{steamid}_ground_truth.csv")
    df_rated = df_gt[df_gt['status'] == 'rated'].dropna(subset=['actual_rating']).copy()
    y = df_rated['actual_rating'].values
    user_appids = df_rated['appid'].values
    
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'difficulty_z'])
    appid_to_idx = {int(aid): i for i, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    y = y[[aid in appid_to_idx for aid in user_appids]]
    
    diff_z = np.clip(full_metadata.iloc[user_indices]['difficulty_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    # 2. Generate RBF Basis
    centers = np.linspace(-2.5, 2.5, 11)
    sigma = 1.0
    X_rbf = np.zeros((len(diff_z), len(centers)))
    for i, c in enumerate(centers):
        X_rbf[:, i] = np.exp(-0.5 * ((diff_z - c) ** 2) / (sigma ** 2))
        
    # 3. Comparative Regression
    X_linear = diff_z.reshape(-1, 1)
    
    print("\n--- Difficulty Modeling Comparison ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_lin, scores_rbf = [], []
    
    for train_idx, test_idx in kf.split(X_linear):
        m_lin = RidgeCV().fit(X_linear[train_idx], y[train_idx])
        scores_lin.append(m_lin.score(X_linear[test_idx], y[test_idx]))
        m_rbf = LassoCV(cv=3, max_iter=10000).fit(X_rbf[train_idx], y[train_idx])
        scores_rbf.append(m_rbf.score(X_rbf[test_idx], y[test_idx]))
        
    print(f"Linear Difficulty CV R^2: {np.mean(scores_lin):.4f}")
    print(f"RBF Difficulty Basis CV R^2: {np.mean(scores_rbf):.4f}")
    
    # 4. Identify the "Sweet Spot"
    final_model = LassoCV(cv=5).fit(X_rbf, y)
    active_indices = np.where(abs(final_model.coef_) > 1e-5)[0]
    
    print("\n--- Identified Difficulty Sweet Spots ---")
    if len(active_indices) == 0:
        print("  - No significant non-linear sweet spot found.")
    else:
        for idx in active_indices:
            print(f"  - Center {centers[idx]:+.1f} Z: Weight {final_model.coef_[idx]:+.4f}")

if __name__ == "__main__":
    research_difficulty_rbf()
