import pandas as pd
import numpy as np
import os

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))

def find_neva():
    idx1 = df[df['name'] == 'Spiritfarer®: Farewell Edition'].index[0]
    idx2 = df[df['name'] == 'Neva'].index[0]
    
    tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
    tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
    sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
    sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
    topic_dist = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_distributions.npy'), mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))

    # Calculate for ALL
    t_fav = tag_vectors[idx1]
    all_t_sims = np.dot(tag_vectors, t_fav) / ((tag_norms + 1.0) * (tag_norms[idx1] + 1.0))
    s_fav = sem_vectors[idx1]
    all_s_sims = np.dot(sem_vectors, s_fav) / ((sem_norms + 1.0) * (sem_norms[idx1] + 1.0))
    
    # Topic Rescue Logic
    fav_topic_z = (topic_dist[idx1] - t_means) / (t_stds + 1e-9)
    fav_topic_z[fav_topic_z < 2.5] = 0
    fav_unit = fav_topic_z / (np.linalg.norm(fav_topic_z) + 1e-9)
    
    fav_topic_z_r = (topic_dist[idx1] - t_means) / (t_stds + 1e-9)
    fav_topic_z_r[fav_topic_z_r < 1.5] = 0
    fav_unit_r = fav_topic_z_r / (np.linalg.norm(fav_topic_z_r) + 1e-9)

    top_sims = np.zeros(len(df))
    batch_size = 50000
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        batch_z = (topic_dist[i:end].astype(np.float32) - t_means) / (t_stds + 1e-9)
        
        # Base
        bz_b = batch_z.copy(); bz_b[bz_b < 2.5] = 0
        norms_b = np.linalg.norm(bz_b, axis=1, keepdims=True) + 1e-9
        sims_b = np.dot(bz_b / norms_b, fav_unit)
        
        # Rescue
        bz_r = batch_z.copy(); bz_r[bz_r < 1.5] = 0
        norms_r = np.linalg.norm(bz_r, axis=1, keepdims=True) + 1e-9
        sims_r = np.dot(bz_r / norms_r, fav_unit_r)
        
        rescue_mask = (np.maximum(all_t_sims[i:end], 0) + np.maximum(all_s_sims[i:end], 0)) > 0.05
        top_sims[i:end] = np.where(rescue_mask, np.maximum(sims_b, sims_r), sims_b)

    def softmin_blend(signals, T=0.01):
        stack = np.stack(signals, axis=0)
        scaled = -stack / T
        max_val = np.max(scaled, axis=0)
        exp_vals = np.exp(scaled - max_val)
        weights = exp_vals / np.sum(exp_vals, axis=0)
        return np.sum(stack * weights, axis=0)

    consensus = softmin_blend([all_t_sims, all_s_sims, top_sims * 0.1])
    # Thematic Blend
    pure = (all_t_sims * 0.25 + all_s_sims * 0.25 + consensus * 0.5)
    
    # Sort
    rank = np.argsort(-pure)
    pos = np.where(rank == idx2)[0][0]
    print(f"Neva Rank: {pos}")
    print(f"Neva Pure Sim: {pure[idx2]}")
    
    # Top 5
    for i in range(5):
        print(f"Top {i+1}: {df.iloc[rank[i]]['name']} ({pure[rank[i]]})")

find_neva()
