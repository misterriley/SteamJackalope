import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR, EMBEDDINGS_DESC_FILE, 
    EMBEDDINGS_DESC_NORMS_FILE, TOPIC_DISTRIBUTIONS_FILE,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, calculate_jackalope_kernel_2d, MIGS, normalize_string

def run_battle():
    sid = '76561198039155404'
    df_gt = pd.read_csv(f'data/user_{sid}_ground_truth.csv')
    df_gt = df_gt[df_gt['status'] == 'rated'].dropna(subset=['actual_rating'])
    y = df_gt['actual_rating'].values
    y_dev = y - 5.0
    
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in df_gt['appid'] if aid in appid_to_idx]
    user_meta_df = full_metadata.iloc[user_indices].copy()
    N = len(user_indices)

    # Load All Components
    print("Loading vectors...")
    g_vecs = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r')[user_indices].astype(np.float32)
    verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r')[user_indices].astype(np.float32)
    sem_vecs = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[user_indices].astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

    # MIGs
    mig_mask_array = np.zeros((len(full_metadata), len(MIGS)), dtype=bool)
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for j, (group, tags) in enumerate(MIGS.items()):
        for t in tags:
            mig_mask_array[:, j] |= tag_series_full.str.contains(f"'{t}':").values
    user_mig_masks = mig_mask_array[user_indices].astype(float)

    from common.utils import extract_seed_metadata
    seed_meta = extract_seed_metadata(user_indices, full_metadata)

    # 1. Similarity Matrices
    print("Calculating similarity matrices...")
    K = calculate_jackalope_kernel_2d(
        verb_profiles, verb_profiles, sem_vecs, sem_norms, sem_vecs, sem_norms,
        topic_dist, topic_dist, t_means, t_stds,
        mig_mask_array[user_indices], mig_mask_array[user_indices],
        user_meta_df['difficulty_z'].values, user_meta_df['difficulty_z'].values,
        user_meta_df['tone_z'].values, user_meta_df['tone_z'].values,
        seed_tags=seed_meta['soul_tags_list'], seed_migs=seed_meta['migs_list'],
        mature_content_flags=seed_meta['mature_flags'], seed_mature_content_flags=seed_meta['mature_flags'],
        graph_embeddings=g_vecs, seed_graph_vecs=g_vecs
    )
    
    G = np.dot(g_vecs, g_vecs.T)
    g_norms = np.linalg.norm(g_vecs, axis=1)
    G = G / (g_norms[:, None] * g_norms[None, :] + 1e-9)
    G = np.maximum(0, G)

    # Leak Mask
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

    # Metadata
    quality_grid = np.load(os.path.join(PRODUCTION_DATA_DIR, "quality_scores_grid.npy"), mmap_mode='r')
    q_feat = to_z(quality_grid[0][user_indices], clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)).reshape(-1, 1)
    meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z', 'tone_z']
    X_meta = np.zeros((N, len(meta_cols)))
    for j, col in enumerate(meta_cols):
        X_meta[:, j] = to_z(user_meta_df[col].values, clamp=(Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX))

    # Battle Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 6, 50)
    
    scenarios = [
        ("CORE 9 (Kernel, Graph, Meta)", [q_feat, X_meta]),
        ("STRUCTURAL 43 (Core + MIGs)", [q_feat, X_meta, user_mig_masks]),
        ("FULL 292 (Structural + Topics)", [q_feat, X_meta, user_mig_masks, topic_dist]),
        ("MINIMAL 2 (Kernel, Graph Only)", [])
    ]

    for name, static_list in scenarios:
        X_static = np.hstack(static_list) if static_list else None
        alpha_scores = np.zeros(len(alphas))
        
        for train_idx, test_idx in kf.split(range(N)):
            # Kernel signal
            K_sub = np.exp(K * 10.0) * lp_mask
            K_train = K_sub[np.ix_(train_idx, train_idx)]
            np.fill_diagonal(K_train, 0.0)
            X_k_train = (np.sum(K_train * y_dev[train_idx], axis=1) / (np.sum(K_train, axis=1) + 1e-9)).reshape(-1, 1)
            
            K_test = K_sub[np.ix_(test_idx, train_idx)]
            X_k_test = (np.sum(K_test * y_dev[train_idx], axis=1) / (np.sum(K_test, axis=1) + 1e-9)).reshape(-1, 1)
            
            # Graph signal
            G_sub = G * lp_mask
            G_train = G_sub[np.ix_(train_idx, train_idx)]
            np.fill_diagonal(G_train, 0.0)
            X_g_train = (np.sum(G_train * y_dev[train_idx], axis=1) / (np.sum(G_train, axis=1) + 1e-9)).reshape(-1, 1)
            
            G_test = G_sub[np.ix_(test_idx, train_idx)]
            X_g_test = (np.sum(G_test * y_dev[train_idx], axis=1) / (np.sum(G_test, axis=1) + 1e-9)).reshape(-1, 1)
            
            X_tr = np.hstack([X_k_train, X_g_train, X_static[train_idx]]) if X_static is not None else np.hstack([X_k_train, X_g_train])
            X_te = np.hstack([X_k_test, X_g_test, X_static[test_idx]]) if X_static is not None else np.hstack([X_k_test, X_g_test])
            
            for i, a in enumerate(alphas):
                ridge = Ridge(alpha=a).fit(X_tr, y[train_idx])
                alpha_scores[i] += ridge.score(X_te, y[test_idx])
        
        best_r2 = np.max(alpha_scores) / 5.0
        best_a = alphas[np.argmax(alpha_scores)]
        print(f"{name}: R2={best_r2:.4f} (Alpha={best_a:.2f})")

if __name__ == '__main__':
    run_battle()
