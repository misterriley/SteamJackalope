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
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)
from common.utils import calculate_jackalope_kernel, MIGS

def check_stellar_nier():
    nier_appid = 524220
    stellar_appid = 3489700
    
    print("Loading metadata and vectors...")
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    
    if nier_appid not in appid_to_idx or stellar_appid not in appid_to_idx:
        print("Error: One of the AppIDs not found.")
        return
        
    n_idx = appid_to_idx[nier_appid]
    s_idx = appid_to_idx[stellar_appid]
    
    # Load Vectors
    all_verbs = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r').astype(np.float32)
    all_sem = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r').astype(np.float32)
    all_sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').astype(np.float32)
    all_topics = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r').astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)
    all_graph = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)

    # Masks
    mig_masks_1d = {}
    tag_series_full = full_metadata['tags'].fillna('').astype(str)
    for group, tags in MIGS.items():
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            if t not in mig_masks_1d:
                mig_masks_1d[t] = tag_series_full.str.contains(pattern, regex=True).values

    from common.utils import extract_seed_metadata
    seed_meta = extract_seed_metadata([n_idx], full_metadata)

    # Calculate Similarity
    print(f"Calculating similarity between {full_metadata.iloc[n_idx]['name']} and {full_metadata.iloc[s_idx]['name']}...")
    
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
    
    s_sim = sims[s_idx]
    print(f"\n>>> FINAL SIMILARITY: {s_sim:.4f} <<<")
    
    # Check if it was in the top 100
    top_100_indices = np.argsort(-sims)[:100]
    is_in = s_idx in top_100_indices
    rank = np.where(top_100_indices == s_idx)[0]
    
    if is_in:
        print(f"Stellar Blade is in the Top 100 (Rank: {rank[0] + 1})")
    else:
        print(f"Stellar Blade is NOT in the Top 100 (Current Rank: {np.where(np.argsort(-sims) == s_idx)[0][0] + 1})")

if __name__ == '__main__':
    check_stellar_nier()
