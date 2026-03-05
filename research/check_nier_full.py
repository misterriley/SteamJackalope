import pandas as pd
import numpy as np
import os
import json
import sys
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR, EMBEDDINGS_DESC_FILE, 
    EMBEDDINGS_DESC_NORMS_FILE, TOPIC_DISTRIBUTIONS_FILE,
    Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)
from common.utils import to_z, calculate_jackalope_kernel, MIGS

def analyze_nier_cluster():
    sid = '76561198039155404'
    nier_appid = 524220
    
    print("Loading metadata and vectors...")
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    
    if nier_appid not in appid_to_idx:
        print(f"Error: AppID {nier_appid} not found in metadata.")
        return
        
    n_idx = appid_to_idx[nier_appid]
    
    # Load Necessary Vectors for Kernel
    all_verbs = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r').astype(np.float32)
    all_sem = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r').astype(np.float32)
    all_sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').astype(np.float32)
    all_topics = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r').astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)
    
    # Structural Masks for Kernel (1D)
    mig_masks_1d = {}
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for group, tags in MIGS.items():
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            if t not in mig_masks_1d:
                mig_masks_1d[t] = tag_series_full.str.contains(pattern, regex=True).values
            
    # Seed Metadata for Kernel
    from common.utils import extract_seed_metadata
    seed_meta = extract_seed_metadata([n_idx], full_metadata)
    
    # 1. Calculate Similarity to NieR: Automata
    print(f"Calculating similarity for {full_metadata.iloc[n_idx]['name']}...")
    all_graph = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)
    
    sims = calculate_jackalope_kernel(
        verb_profiles=all_verbs, seed_verb_profile=all_verbs[n_idx],
        sem_vectors=all_sem, sem_norms=all_sem_norms, seed_sem_vec=all_sem[n_idx], seed_sem_norm=all_sem_norms[n_idx],
        topic_distributions=all_topics, seed_topic_dist=all_topics[n_idx],
        topic_means=t_means, topic_stds=t_stds,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
        seed_migs=seed_meta['migs_list'][0], seed_tags=seed_meta['soul_tags_list'][0],
        candidate_anchor_masks=mig_masks_1d,
        difficulty_z=full_metadata['difficulty_z'].values, seed_difficulty_z=full_metadata.iloc[n_idx]['difficulty_z'],
        tone_z=full_metadata['tone_z'].values, seed_tone_z=full_metadata.iloc[n_idx]['tone_z'],
        graph_embeddings=all_graph, seed_graph_vec=all_graph[n_idx]
    )
    
    # Get top 100 most similar (excluding itself)
    sims[n_idx] = -1
    top_100_indices = np.argsort(-sims)[:100]
    
    # 2. Rank these 100 by User Taste DNA
    print("Loading User Taste DNA (Core 9 Mode)...")
    ratings_path = f'data/user_{sid}_predicted_ratings.npy'
    if not os.path.exists(ratings_path):
        print("Error: Predicted ratings not found. Run solver first.")
        return
    user_scores = np.load(ratings_path)
    
    results = []
    for idx in top_100_indices:
        row = full_metadata.iloc[idx]
        results.append({
            'appid': int(row['appid']),
            'name': row['name'],
            'sim_score': float(sims[idx]),
            'user_score': float(user_scores[idx])
        })
        
    # Sort by user_score
    ranked_results = sorted(results, key=lambda x: x['user_score'], reverse=True)
    
    print(f"\n=== THE NIER: AUTOMATA CLUSTER (Top 100 Sim) ranked by Taste DNA ===")
    print(f"{'Game Name':<40} | {'Sim':<6} | {'User Rating':<12}")
    print("-" * 65)
    for i, res in enumerate(ranked_results[:50]):
        print(f"{res['name'][:39]:<40} | {res['sim_score']:.3f} | {res['user_score']:.2f}")

    # Identify the "Bottom" of the 100 - these are the "weird/bad" ones the user doesn't like
    print("\n--- WEIRDEST/WORST MATCHES (Low User Score, High NieR Sim) ---")
    for i, res in enumerate(ranked_results[-10:]):
        print(f"{res['name'][:39]:<40} | {res['sim_score']:.3f} | {res['user_score']:.2f}")

if __name__ == '__main__':
    analyze_nier_cluster()
