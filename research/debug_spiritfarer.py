import pandas as pd
import numpy as np
import json
import os
import re

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))

def softmin_blend(signals, T=0.01):
    stack = np.stack(signals)
    scaled = -stack / T
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals)
    return np.sum(stack * weights)

def debug_seed(seed_name):
    idx1 = df[df['name'].str.lower() == seed_name.lower()].index[0]
    
    tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
    tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
    sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
    sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
    topic_dist = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_distributions.npy'), mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))

    # Pre-calculate for all
    t_fav = tag_vectors[idx1]
    t_norm_fav = tag_norms[idx1]
    all_t_sims = np.dot(tag_vectors, t_fav) / ((tag_norms + 1.0) * (t_norm_fav + 1.0))
    
    s_fav = sem_vectors[idx1]
    s_norm_fav = sem_norms[idx1]
    all_s_sims = np.dot(sem_vectors, s_fav) / ((sem_norms + 1.0) * (s_norm_fav + 1.0))
    
    # Topic Sim
    def get_all_topic_sims(threshold):
        fav_topic_z = (topic_dist[idx1] - t_means) / (t_stds + 1e-9)
        fav_topic_z[fav_topic_z < threshold] = 0 
        fav_topic_z_unit = fav_topic_z / (np.linalg.norm(fav_topic_z) + 1e-9)
        
        sims = np.zeros(len(df), dtype=np.float32)
        batch_size = 50000
        for i in range(0, len(df), batch_size):
            end = min(i + batch_size, len(df))
            batch_z = (topic_dist[i:end].astype(np.float32) - t_means) / (t_stds + 1e-9)
            batch_z[batch_z < threshold] = 0
            batch_norms = np.linalg.norm(batch_z, axis=1, keepdims=True) + 1e-9
            sims[i:end] = np.dot(batch_z / batch_norms, fav_topic_z_unit)
        return sims

    top_sims_base = get_all_topic_sims(2.5)
    rescue_mask = (np.maximum(all_t_sims, 0) + np.maximum(all_s_sims, 0)) > 0.05
    top_sims_rescue = get_all_topic_sims(1.5)
    top_sims = top_sims_base
    top_sims[rescue_mask] = np.maximum(top_sims[rescue_mask], top_sims_rescue[rescue_mask])
    
    # Final Combine
    res = []
    for i in range(len(df)):
        if i == idx1: continue
        c_sim = softmin_blend([all_t_sims[i], all_s_sims[i], top_sims[i]*0.1], 0.01)
        thematic_rescue = (max(all_t_sims[i], 0) + max(all_s_sims[i], 0)) / 2.0
        c_sim = max(c_sim, thematic_rescue * 0.2)
        
        pure_thematic = (all_t_sims[i] + c_sim) / 2.0
        
        if pure_thematic > 0.02:
            res.append({
                "name": df.iloc[i]['name'],
                "tag": float(all_t_sims[i]),
                "sem": float(all_s_sims[i]),
                "topic": float(top_sims[i]),
                "consensus": float(c_sim),
                "pure_thematic": float(pure_thematic)
            })
            
    res = sorted(res, key=lambda x: x['pure_thematic'], reverse=True)
    return res[:10]

print(json.dumps(debug_seed("Spiritfarer®: Farewell Edition"), indent=2))
