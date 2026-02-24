import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LassoCV

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    METADATA_FILE, QUALITY_GRID_FILE, TOPIC_DISTRIBUTIONS_FILE, 
    PRODUCTION_DATA_DIR, TOPIC_GLOBAL_SCALING_FACTOR,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)

def test_topic_only(user_id="76561198039155404", sample_size=100):
    print("--- Topic-Only Solver Test: N=" + str(sample_size) + " ---")
    
    gt_path = "data/user_" + user_id + "_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    df_sub = df_gt.sample(n=sample_size, random_state=42)
    
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z'])
    appid_to_idx = {aid: i for i, aid in enumerate(df_meta['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_sub['appid'] if aid in appid_to_idx]
    y = df_sub['actual_rating'].values[:len(user_indices)]
    
    q_feat = np.clip(np.load(QUALITY_GRID_FILE, mmap_mode='r')[10][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    meta_feat = np.clip(df_meta.iloc[user_indices][['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy"))
    topic_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy"))
    topic_std = (topic_dist - topic_means) / (topic_stds + 1e-10)
    topic_feat = topic_std * TOPIC_GLOBAL_SCALING_FACTOR
    
    X = np.hstack([q_feat.reshape(-1, 1), meta_feat, topic_feat])
    
    model = LassoCV(cv=5, max_iter=20000)
    model.fit(X, y)
    
    coeffs = model.coef_
    r2 = model.score(X, y)
    
    print("Model R^2: " + f"{r2:.4f}")
    print("Alpha:     " + f"{model.alpha_:.4f}")
    
    topic_coeffs = coeffs[6:]
    topic_norm = np.linalg.norm(topic_coeffs)
    
    print("Topic Match Weight (Norm): " + f"{topic_norm:.4f}")
    
    top_t_idx = np.argsort(-np.abs(topic_coeffs))[:5]
    print("\n--- Top 5 Predictive Topics ---")
    for idx in top_t_idx:
        if abs(topic_coeffs[idx]) > 1e-6:
            print("Topic " + str(idx) + ": Weight=" + f"{topic_coeffs[idx]:.4f}")

if __name__ == "__main__":
    test_topic_only()
