import pandas as pd
import numpy as np
import os
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, PRODUCTION_DATA_DIR, TOPIC_DISTRIBUTIONS_FILE
from common.utils import calculate_jackalope_kernel, MIGS, extract_seed_metadata

def verify_kernel_scores():
    meta = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): i for i, aid in enumerate(meta['appid'])}
    
    v_idx = appid_to_idx[1794680] # Vampire Survivors
    o_idx = appid_to_idx[2133330] # The Otter Ways
    h_idx = appid_to_idx[2218750] # Halls of Torment
    
    verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, 'diffused_verb_profiles.npy'), mmap_mode='r')
    sem_vecs = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
    sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
    topic_dists = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))
    g_vecs = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r')
    
    anchor_masks = {}
    tag_series = meta['tags'].fillna('').astype(str)
    all_needed_tags = set()
    for group_tags in MIGS.values(): all_needed_tags.update(group_tags)
    for t in all_needed_tags:
        pattern = rf"'{re.escape(t)}':"
        anchor_masks[t] = tag_series.str.contains(pattern, regex=True).values
        
    seed_meta = extract_seed_metadata([v_idx], meta)
    
    # Load Real MIG Masks
    mig_mask_array = np.zeros((len(meta), len(MIGS)), dtype=bool)
    tag_series_full = meta['tags'].fillna('').astype(str)
    for j, (group, tags) in enumerate(MIGS.items()):
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            mig_mask_array[:, j] |= tag_series_full.str.contains(pattern, regex=True).values

    # CALCULATE RAW KERNEL SCORES using 2D logic
    from common.utils import calculate_jackalope_kernel_2d
    
    def debug_k(idx, name):
        from common.utils import extract_seed_metadata, MIGS, MIG_WEIGHTS
        seed_meta = extract_seed_metadata([v_idx], meta)
        
        # Manually calculate components to see where it breaks
        v_c = verb_profiles[idx:idx+1].astype(np.float32)
        v_s = verb_profiles[v_idx:v_idx+1].astype(np.float32)
        tag_sim = np.sum(np.minimum(v_c, v_s), axis=1) / (np.sum(np.maximum(v_c, v_s), axis=1) + 1e-9)
        
        c_mig = mig_mask_array[idx:idx+1].astype(np.float32)
        s_mig = mig_mask_array[v_idx:v_idx+1].astype(np.float32)
        w_vec = np.array([MIG_WEIGHTS.get(g, 1.0) for g in MIGS.keys()], dtype=np.float32)
        inter_w = np.dot(c_mig * w_vec, s_mig.T)
        union_w = np.dot(c_mig, w_vec)[:, None] + np.dot(s_mig, w_vec)[None, :] - inter_w
        id_match = inter_w / (union_w + 1e-9)
        
        s_dot = np.dot(sem_vecs[idx:idx+1].astype(np.float32), sem_vecs[v_idx:v_idx+1].astype(np.float32).T)
        s_sim = (s_dot / (sem_norms[idx:idx+1, None] + 0.5)) / (sem_norms[v_idx:v_idx+1, None].T + 0.5)
        
        t_sim = np.dot(topic_dists[idx:idx+1].astype(np.float32), topic_dists[v_idx:v_idx+1].astype(np.float32).T)
        
        print(f"\n--- Debug: {name} ---")
        print(f"Tag Sim:      {tag_sim[0]:.4f}")
        print(f"ID Match:     {id_match[0,0]:.4f}")
        print(f"Semantic Sim: {s_sim[0,0]:.4f}")
        print(f"Topic Sim:    {t_sim[0,0]:.4f}")
        
        k = calculate_jackalope_kernel_2d(
            verb_profiles=verb_profiles[idx:idx+1], seed_verb_profiles=verb_profiles[v_idx:v_idx+1],
            sem_vectors=sem_vecs[idx:idx+1], sem_norms=sem_norms[idx:idx+1], 
            seed_sem_vecs=sem_vecs[v_idx:v_idx+1], seed_sem_norms=sem_norms[v_idx:v_idx+1],
            topic_distributions=topic_dists[idx:idx+1], seed_topic_dists=topic_dists[v_idx:v_idx+1],
            topic_means=t_means, topic_stds=t_stds,
            candidate_mig_masks=mig_mask_array[idx:idx+1], 
            seed_mig_masks=mig_mask_array[v_idx:v_idx+1],
            difficulty_z=meta['difficulty_z'].values[idx:idx+1], seed_difficulty_z=meta['difficulty_z'].values[v_idx:v_idx+1],
            tone_z=meta['tone_z'].values[idx:idx+1], seed_tone_z=meta['tone_z'].values[v_idx:v_idx+1],
            seed_tags=[seed_meta['soul_tags_list'][0]], seed_migs=[seed_meta['migs_list'][0]],
            mature_content_flags=meta['mature_content'].values[idx:idx+1] > 0,
            seed_mature_content_flags=meta['mature_content'].values[v_idx:v_idx+1] > 0,
            graph_embeddings=g_vecs[idx:idx+1], seed_graph_vecs=g_vecs[v_idx:v_idx+1]
        )
        print(f"Final Kernel: {k[0, 0]:.6f}")

    debug_k(h_idx, "Halls of Torment")
    debug_k(o_idx, "The Otter Ways")

if __name__ == '__main__':
    verify_kernel_scores()
