import pandas as pd
import numpy as np
import json
import os
import re
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR, 
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE, 
    TOPIC_DISTRIBUTIONS_FILE, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, calculate_jackalope_kernel, MIGS, extract_seed_metadata

def benchmark_similarity():
    sid = '76561198039155404'
    print(f"Loading environment for user {sid}...")
    
    meta = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): i for i, aid in enumerate(meta['appid'])}
    
    # Seed: Vampire Survivors
    seed_appid = 1794680
    seed_idx = appid_to_idx[seed_appid]
    
    # Filters
    gt = pd.read_csv(f'data/user_{sid}_ground_truth.csv')
    rated_appids = set(gt[gt['status'] == 'rated']['appid'].tolist())
    rated_indices = [appid_to_idx[aid] for aid in rated_appids if aid in appid_to_idx]
    
    non_rated_mask = np.ones(len(meta), dtype=bool)
    non_rated_mask[rated_indices] = False
    non_rated_mask[seed_idx] = False
    
    # User Profile & Scores
    pred_scores = np.load(f'data/user_{sid}_predicted_ratings.npy')
    
    # 1. GRAPH SIMILARITY
    g_vecs = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r')
    seed_g = g_vecs[seed_idx]
    g_dot = np.dot(g_vecs, seed_g)
    g_norms = np.linalg.norm(g_vecs, axis=1) * np.linalg.norm(seed_g)
    g_sims = g_dot / (g_norms + 1e-9)
    
    # 2. KERNEL SIMILARITY
    print("Calculating Kernel similarities...")
    verb_profiles = np.load(os.path.join(PRODUCTION_DATA_DIR, 'diffused_verb_profiles.npy'), mmap_mode='r')
    sem_vecs = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_dists = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))
    
    # Anchor Masks
    anchor_masks = {}
    tag_series = meta['tags'].fillna('').astype(str)
    all_needed_tags = set()
    for tags in MIGS.values(): all_needed_tags.update(tags)
    for t in all_needed_tags:
        pattern = rf"'{re.escape(t)}':"
        anchor_masks[t] = tag_series.str.contains(pattern, regex=True).values
        
    seed_meta = extract_seed_metadata([seed_idx], meta)
    
    k_sims = calculate_jackalope_kernel(
        verb_profiles=verb_profiles, seed_verb_profile=verb_profiles[seed_idx],
        sem_vectors=sem_vecs, sem_norms=sem_norms, seed_sem_vec=sem_vecs[seed_idx], seed_sem_norm=sem_norms[seed_idx],
        topic_distributions=topic_dists, seed_topic_dist=topic_dists[seed_idx],
        topic_means=t_means, topic_stds=t_stds,
        tag_scaling_factor=11.25, dot_product_lambda=0.5,
        sem_scaling_factor=11.25, sem_lambda=0.5,
        mature_content_flags=meta['mature_content'].values > 0,
        seed_mature_content=seed_meta['mature_flags'][0],
        seed_migs=seed_meta['migs_list'][0],
        seed_tags=seed_meta['soul_tags_list'][0],
        candidate_anchor_masks=anchor_masks,
        active_narrative_seed=seed_meta['active_narrative'],
        is_cinematic_seed=seed_meta['is_cinematic'],
        difficulty_z=meta['difficulty_z'].values, seed_difficulty_z=meta['difficulty_z'].values[seed_idx],
        tone_z=meta['tone_z'].values, seed_tone_z=meta['tone_z'].values[seed_idx],
        graph_embeddings=g_vecs, seed_graph_vec=seed_g
    )
    
    # 3. ENSEMBLE (Kernel-First Fix)
    pop_z = to_z(meta['pop_z'].values)
    g_spec = g_sims / (np.maximum(0, pop_z) + 1.0)
    t_sims = np.dot(topic_dists, topic_dists[seed_idx])
    
    # Kernel-First weighting to prioritize mechanical fidelity
    ensemble = 0.8 * k_sims + 0.15 * g_spec + 0.05 * t_sims
    
    # --- REPORTING ---
    metrics = [
        ('RAW GRAPH', g_sims),
        ('PURE KERNEL', k_sims),
        ('ALL-IN ENSEMBLE', ensemble)
    ]
    
    for name, scores_sim in metrics:
        scores_sim_clean = scores_sim.copy()
        scores_sim_clean[~non_rated_mask] = -1e12
        top_100_idx = np.argsort(-scores_sim_clean)[:100]
        
        # Rank by User Score
        final_rank = np.argsort(-pred_scores[top_100_idx])[:10]
        
        print(f"\n--- TOP 10 (METRIC: {name}) ---")
        for i in final_rank:
            idx = top_100_idx[i]
            print(f"{meta.iloc[idx]['name']} (Sim: {scores_sim[idx]:.3f}, User Score: {pred_scores[idx]:.2f})")

if __name__ == '__main__':
    benchmark_similarity()
