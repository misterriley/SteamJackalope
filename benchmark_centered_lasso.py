import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
import os
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, PRODUCTION_DATA_DIR,
    TAG_NORMS_FILE, EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, calculate_jackalope_kernel_2d, MIGS, normalize_string

def benchmark_centered(user_id="76561198039155404"):
    gt_path = f"data/user_{user_id}_ground_truth.csv"
    df_gt = pd.read_csv(gt_path).dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices]
    N = len(user_indices)

    # --- FEATURE ASSEMBLY ---
    all_graph_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)
    user_graph_vectors = all_graph_vectors[user_indices]
    
    verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r')[user_indices].astype(np.float32)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

    mig_mask_array = np.zeros((len(full_metadata), len(MIGS)), dtype=bool)
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for j, (group, tags) in enumerate(MIGS.items()):
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            mig_mask_array[:, j] |= tag_series_full.str.contains(pattern, regex=True).values
    user_mig_masks = mig_mask_array[user_indices]

    from common.utils import extract_seed_metadata
    seed_meta = extract_seed_metadata(user_indices, full_metadata)
    
    K_train = calculate_jackalope_kernel_2d(
        verb_profiles=verb_profiles, seed_verb_profiles=verb_profiles,
        sem_vectors=sem_vectors, sem_norms=sem_norms, seed_sem_vecs=sem_vectors, seed_sem_norms=sem_norms,
        topic_distributions=topic_dist, seed_topic_dists=topic_dist,
        topic_means=t_means, topic_stds=t_stds,
        candidate_mig_masks=user_mig_masks, seed_mig_masks=user_mig_masks,
        difficulty_z=user_meta_df['difficulty_z'].values, seed_difficulty_z=user_meta_df['difficulty_z'].values,
        tone_z=user_meta_df['tone_z'].values, seed_tone_z=user_meta_df['tone_z'].values,
        seed_tags=seed_meta['soul_tags_list'], seed_migs=seed_meta['migs_list'],
        mature_content_flags=seed_meta['mature_flags'], seed_mature_content_flags=seed_meta['mature_flags'],
        graph_embeddings=user_graph_vectors, seed_graph_vecs=user_graph_vectors
    )

    user_names = user_meta_df['name'].tolist()
    leak_proof_mask = (K_train < 0.95)
    for i in range(N):
        clean_i = set(normalize_string(user_names[i]).split())
        if len(clean_i) < 2: continue 
        for j in range(i + 1, N):
            clean_j = set(normalize_string(user_names[j]).split())
            overlap = clean_i.intersection(clean_j)
            if len(overlap) >= 2 and (len(overlap) / max(len(clean_i), len(clean_j)) > 0.7):
                leak_proof_mask[i, j] = False
                leak_proof_mask[j, i] = False

    K_exp = np.exp(K_train * 10.0)
    K_exp *= leak_proof_mask
    np.fill_diagonal(K_exp, 0.0)
    y_dev = y - 5.0
    X_kernel = (np.sum(K_exp * y_dev, axis=1) / (np.sum(K_exp, axis=1) + 1e-9)).reshape(-1, 1)

    X_mig = user_mig_masks.astype(float)

    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_all = [np.corrcoef(to_z(quality_grid[k][user_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)), y)[0, 1] if np.std(quality_grid[k][user_indices]) > 1e-9 else 0 for k in range(quality_grid.shape[0])]
    best_q_idx = np.argmax(np.abs(q_all))
    q_global = to_z(quality_grid[best_q_idx][user_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))
    
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z', 'tone_z']
    X_meta = np.zeros((user_indices.__len__(), len(meta_cols)))
    for j, col in enumerate(meta_cols):
        X_meta[:, j] = to_z(user_meta_df[col].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))

    dot_graph = np.dot(user_graph_vectors, user_graph_vectors.T)
    g_norms = np.linalg.norm(user_graph_vectors, axis=1)
    graph_sim_matrix = dot_graph / (g_norms[:, None] * g_norms[None, :] + 1e-9)
    graph_sim_matrix = np.maximum(0, graph_sim_matrix)
    graph_sim_matrix *= leak_proof_mask
    np.fill_diagonal(graph_sim_matrix, 0.0)
    X_graph = (np.dot(graph_sim_matrix, y_dev) / (np.sum(graph_sim_matrix, axis=1) + 1e-9)).reshape(-1, 1)

    X = np.hstack([X_kernel, X_graph, q_global.reshape(-1, 1), X_meta, X_mig])

    # --- BENCHMARK: Centered vs Non-Centered ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Non-Centered (Current Baseline)
    nc_scores = []
    for train_idx, test_idx in kf.split(X):
        model = LassoCV(cv=5, max_iter=10000).fit(X[train_idx], y[train_idx])
        nc_scores.append(model.score(X[test_idx], y[test_idx]))
    
    # 2. Centered
    X_means = np.mean(X, axis=0)
    X_centered = X - X_means
    c_scores = []
    for train_idx, test_idx in kf.split(X_centered):
        model = LassoCV(cv=5, max_iter=10000).fit(X_centered[train_idx], y[train_idx])
        c_scores.append(model.score(X_centered[test_idx], y[test_idx]))

    # 3. Z-Scored (Centered + Scaled)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    s_scores = []
    for train_idx, test_idx in kf.split(X_scaled):
        model = LassoCV(cv=5, max_iter=10000).fit(X_scaled[train_idx], y[train_idx])
        s_scores.append(model.score(X_scaled[test_idx], y[test_idx]))

    # Final Train on full scaled data
    model_scaled = LassoCV(cv=5, max_iter=10000).fit(X_scaled, y)
    
    print(f"\n--- Lasso Normalization Benchmark ---")
    print(f"Non-Centered OOS R2: {np.mean(nc_scores):.4f}")
    print(f"Centered OOS R2:     {np.mean(c_scores):.4f}")
    print(f"Z-Scored OOS R2:     {np.mean(s_scores):.4f}")
    
    print(f"\n--- Z-Scored Coefficients (First 10) ---")
    print(f"Intercept (Mean Rating): {model_scaled.intercept_:.4f}")
    print(f"Kernel Match Coeff:      {model_scaled.coef_[0]:.4f}")
    print(f"Graph Match Coeff:       {model_scaled.coef_[1]:.4f}")
    print(f"Quality Coeff:           {model_scaled.coef_[2]:.4f}")

if __name__ == "__main__":
    benchmark_centered()
