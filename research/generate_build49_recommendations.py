import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LassoCV

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, QUALITY_GRID_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, TAG_NORMS_FILE, 
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)

def generate_recommendations(user_id="76561198039155404"):
    print("--- Generating Top Recommendations (Build 49 Engine) ---")
    
    df_meta = pd.read_parquet(METADATA_FILE)
    num_games = len(df_meta)
    
    gt_path = "data/user_" + user_id + "_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    n_samples = len(y)
    
    appid_to_idx = {aid: i for i, aid in enumerate(df_meta['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    y = y[[aid in appid_to_idx for aid in df_gt['appid']]] 
    
    W_TAG = 1.0
    W_SEM = 2.0
    W_TOP = 26.5
    
    print("Preparing training features...")
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r').reshape(-1, 1)
    tag_feat_user = (tag_vectors[user_indices] / (tag_norms[user_indices] + DOT_PRODUCT_LAMBDA)) * W_TAG
    
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').reshape(-1, 1)
    sem_feat_user = (sem_vectors[user_indices] / (sem_norms[user_indices] + SEMANTIC_DOT_PRODUCT_LAMBDA)) * W_SEM
    
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    topic_feat_user = topic_distributions[user_indices].astype(np.float32) * W_TOP

    all_thematic_user = np.hstack([tag_feat_user, sem_feat_user, topic_feat_user])
    
    allowed = n_samples - 7
    print("Applying Relevance Filter (p <= " + str(allowed) + ")...")
    correlations = []
    for i in range(all_thematic_user.shape[1]):
        feat = all_thematic_user[:, i]
        corr = np.corrcoef(feat, y)[0, 1] if np.std(feat) > 1e-9 else 0.0
        correlations.append(abs(corr))
    
    top_indices = np.argsort(-np.array(correlations))[:allowed]
    
    q_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
    q_feat_user = np.clip(q_grid[10][user_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    meta_feat_user = np.clip(df_meta.iloc[user_indices][['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    X_train = np.hstack([q_feat_user.reshape(-1, 1), meta_feat_user, all_thematic_user[:, top_indices]])
    
    print("Solving LASSO...")
    model = LassoCV(cv=5, max_iter=20000)
    model.fit(X_train, y)
    
    print("Scoring " + str(num_games) + " games...")
    batch_size = 50000
    scores = np.zeros(num_games)
    
    for i in range(0, num_games, batch_size):
        end = min(i + batch_size, num_games)
        
        b_tag = (tag_vectors[i:end] / (tag_norms[i:end] + DOT_PRODUCT_LAMBDA)) * W_TAG
        b_sem = (sem_vectors[i:end] / (sem_norms[i:end] + SEMANTIC_DOT_PRODUCT_LAMBDA)) * W_SEM
        b_top = topic_distributions[i:end].astype(np.float32) * W_TOP
        
        b_thematic = np.hstack([b_tag, b_sem, b_top])[:, top_indices]
        
        b_q = np.clip(q_grid[10][i:end], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
        b_meta = np.clip(df_meta.iloc[i:end][['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
        
        X_batch = np.hstack([b_q.reshape(-1, 1), b_meta, b_thematic])
        scores[i:end] = model.predict(X_batch)

    print("Applying filters and exclusions...")
    mask = np.ones(num_games, dtype=bool)
    mask &= df_meta['is_english'].values.astype(bool)
    mask &= ~df_meta['is_vr_only'].values.astype(bool)
    mask &= ~df_meta['is_delisted'].values.astype(bool)
    mask &= ~df_meta['is_hollow'].values.astype(bool)
    mask[user_indices] = False
    
    scores[~mask] = -1e12
    
    top_20_idx = np.argsort(-scores)[:20]
    
    print("\n--- TOP 20 RECOMMENDATIONS (Build 49) ---")
    print("{:<4} | {:<6} | {:<}".format("Rank", "Score", "Game Name"))
    print("-" * 50)
    for i, idx in enumerate(top_20_idx):
        print("{:<4} | {:<6.2f} | {:<}".format(i+1, scores[idx], df_meta.iloc[idx]['name']))

if __name__ == "__main__":
    generate_recommendations()
