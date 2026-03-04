import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge
from sklearn.model_selection import KFold
import os
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, calculate_jackalope_kernel_2d, MIGS, normalize_string

def benchmark_all_features(user_id="76561198039155404"):
    gt_path = f"data/user_{user_id}_ground_truth.csv"
    df_gt = pd.read_csv(gt_path)
    # Strictly filter for verified human ratings
    df_gt = df_gt[df_gt['status'] == 'rated'].dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    y_dev_global = y - 5.0
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices]
    N = len(user_indices)

    # --- RAW COMPONENTS ---
    all_graph_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)
    user_graph_vectors = all_graph_vectors[user_indices]
    verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r')[user_indices].astype(np.float32)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    all_topics = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r').astype(np.float32)
    user_topics = all_topics[user_indices]
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

    # 1. Structural Setup
    mig_mask_array = np.zeros((len(full_metadata), len(MIGS)), dtype=bool)
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for j, (group, tags) in enumerate(MIGS.items()):
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            mig_mask_array[:, j] |= tag_series_full.str.contains(pattern, regex=True).values
    user_mig_masks = mig_mask_array[user_indices]

    from common.utils import extract_seed_metadata
    seed_meta = extract_seed_metadata(user_indices, full_metadata)

    # 2. Similarity Matrices (NxN)
    K_full = calculate_jackalope_kernel_2d(
        verb_profiles=verb_profiles, seed_verb_profiles=verb_profiles,
        sem_vectors=sem_vectors, sem_norms=sem_norms, seed_sem_vecs=sem_vectors, seed_sem_norms=sem_norms,
        topic_distributions=topic_dist, seed_topic_dists=topic_dist,
        topic_means=t_means, topic_stds=t_stds,
        candidate_mig_masks=user_mig_masks,
        seed_mig_masks=user_mig_masks,
        difficulty_z=user_meta_df['difficulty_z'].values, seed_difficulty_z=user_meta_df['difficulty_z'].values,
        tone_z=user_meta_df['tone_z'].values, seed_tone_z=user_meta_df['tone_z'].values,
        seed_tags=seed_meta['soul_tags_list'], seed_migs=seed_meta['migs_list'],
        mature_content_flags=seed_meta['mature_flags'], seed_mature_content_flags=seed_meta['mature_flags'],
        graph_embeddings=user_graph_vectors, seed_graph_vecs=user_graph_vectors
    )

    dot_graph = np.dot(user_graph_vectors, user_graph_vectors.T)
    g_norms = np.linalg.norm(user_graph_vectors, axis=1)
    G_full = dot_graph / (g_norms[:, None] * g_norms[None, :] + 1e-9)
    G_full = np.maximum(0, G_full)

    # Leak-Proof Mask
    user_names = user_meta_df['name'].tolist()
    lp_mask = np.ones((N, N), dtype=bool)
    for i in range(N):
        clean_i = set(normalize_string(user_names[i]).split())
        if len(clean_i) < 2: continue 
        for j in range(i + 1, N):
            clean_j = set(normalize_string(user_names[j]).split())
            overlap = clean_i.intersection(clean_j)
            if len(overlap) >= 2 and (len(overlap) / max(len(clean_i), len(clean_j)) > 0.7):
                lp_mask[i, j] = lp_mask[j, i] = False

    # 3. Static Features
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_feat = to_z(quality_grid[0][user_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z', 'tone_z']
    X_meta = np.zeros((N, len(meta_cols)))
    for j, col in enumerate(meta_cols):
        X_meta[:, j] = to_z(user_meta_df[col].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    
    # Static Pool: [Q, Meta, MIGs, Topics]
    X_static = np.hstack([q_feat.reshape(-1, 1), X_meta, user_mig_masks.astype(float), user_topics])

    # --- STRICT OOS LOOP ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 6, 50)
    alpha_scores = np.zeros(len(alphas))
    
    print(f"\n--- 'All-In' discovery Benchmark (Strict OOS) ---")
    print(f"Features: Metadata + Graph + Topics + Kernel ({X_static.shape[1] + 2} total)")
    
    for train_idx, test_idx in kf.split(range(N)):
        # Assembly (Exactly matching production solver logic)
        K_sub_train = K_full[np.ix_(train_idx, train_idx)]
        lp_train = lp_mask[np.ix_(train_idx, train_idx)]
        K_exp_train = np.exp(K_sub_train * 10.0) * lp_train
        np.fill_diagonal(K_exp_train, 0.0)
        X_k_train = (np.sum(K_exp_train * y_dev_global[train_idx], axis=1) / (np.sum(K_exp_train, axis=1) + 1e-9)).reshape(-1, 1)
        
        G_train = G_full[np.ix_(train_idx, train_idx)] * lp_train
        np.fill_diagonal(G_train, 0.0)
        X_g_train = (np.sum(G_train * y_dev_global[train_idx], axis=1) / (np.sum(G_train, axis=1) + 1e-9)).reshape(-1, 1)
        
        X_train = np.hstack([X_k_train, X_g_train, X_static[train_idx]])
        
        K_sub_test = K_full[np.ix_(test_idx, train_idx)]
        lp_test = lp_mask[np.ix_(test_idx, train_idx)]
        K_exp_test = np.exp(K_sub_test * 10.0) * lp_test
        X_k_test = (np.sum(K_exp_test * y_dev_global[train_idx], axis=1) / (np.sum(K_exp_test, axis=1) + 1e-9)).reshape(-1, 1)
        
        G_test = G_full[np.ix_(test_idx, train_idx)] * lp_test
        X_g_test = (np.sum(G_test * y_dev_global[train_idx], axis=1) / (np.sum(G_test, axis=1) + 1e-9)).reshape(-1, 1)
        
        X_test = np.hstack([X_k_test, X_g_test, X_static[test_idx]])
        
        for i, a in enumerate(alphas):
            ridge = Ridge(alpha=a).fit(X_train, y[train_idx])
            alpha_scores[i] += ridge.score(X_test, y[test_idx])

    best_idx = np.argmax(alpha_scores)
    print(f"Ridge OOS R2:       {alpha_scores[best_idx] / 5.0:.4f} (Alpha: {alphas[best_idx]:.4f})")

if __name__ == "__main__":
    benchmark_all_features()
