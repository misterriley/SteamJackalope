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

def calculate_quality(pos, neg):
    """Production quality logic: Bayesian Average + Log Scaling."""
    total = pos + neg
    if total < 1: return 0.5
    raw = pos / total
    # Bayesian Smoothing: Pull towards 0.7 (Average Steam Game)
    smoothed = (pos + 35) / (total + 50)
    # Log-scale boost for high-volume games
    vol_boost = np.log10(total + 1) / 10.0
    return smoothed + vol_boost

def audit_quality_weeding():
    sid = '76561198039155404'
    vs_appid = 1794680
    
    print("Loading metadata and calculating quality scores...")
    df = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(df['appid'])}
    vs_idx = appid_to_idx[vs_appid]
    
    # Pre-calculate Quality Scores for all games
    q_scores = np.array([calculate_quality(p, n) for p, n in zip(df['positive'], df['negative'])])
    q_z = to_z(q_scores) # Standardize for comparison
    
    # Load Necessary Vectors for Kernel
    all_verbs = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r').astype(np.float32)
    all_sem = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r').astype(np.float32)
    all_sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').astype(np.float32)
    all_topics = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r').astype(np.float32)
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)
    
    # Structural Masks for Kernel (1D)
    mig_masks_1d = {}
    tag_series_full = df['tags'].fillna('').astype(str)
    for group, tags in MIGS.items():
        for t in tags:
            pattern = rf"'{re.escape(t)}':"
            if t not in mig_masks_1d:
                mig_masks_1d[t] = tag_series_full.str.contains(pattern, regex=True).values
            
    # Seed Metadata for Kernel
    from common.utils import extract_seed_metadata
    seed_meta = extract_seed_metadata([vs_idx], df)
    
    # 1. Calculate Similarity to Vampire Survivors
    print(f"Calculating similarity for {df.iloc[vs_idx]['name']}...")
    all_graph = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)
    
    sims = calculate_jackalope_kernel(
        verb_profiles=all_verbs, seed_verb_profile=all_verbs[vs_idx],
        sem_vectors=all_sem, sem_norms=all_sem_norms, seed_sem_vec=all_sem[vs_idx], seed_sem_norm=all_sem_norms[vs_idx],
        topic_distributions=all_topics, seed_topic_dist=all_topics[vs_idx],
        topic_means=t_means, topic_stds=t_stds,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
        seed_migs=seed_meta['migs_list'][0], seed_tags=seed_meta['soul_tags_list'][0],
        candidate_anchor_masks=mig_masks_1d,
        difficulty_z=df['difficulty_z'].values, seed_difficulty_z=df.iloc[vs_idx]['difficulty_z'],
        tone_z=df['tone_z'].values, seed_tone_z=df.iloc[vs_idx]['tone_z'],
        graph_embeddings=all_graph, seed_graph_vec=all_graph[vs_idx]
    )
    
    # Get top 100 most similar (excluding itself)
    sims[vs_idx] = -1
    top_100_indices = np.argsort(-sims)[:100]
    
    # Calculate Quality-Weighted Similarity (Experimental)
    # Filter 1: Raw Quality Gate (Z > -1.0)
    # Filter 2: Composite Score (Sim * QualityFactor)
    
    results = []
    for idx in top_100_indices:
        row = df.iloc[idx]
        results.append({
            'appid': int(row['appid']),
            'name': row['name'],
            'sim_score': float(sims[idx]),
            'quality_z': float(q_z[idx]),
            'composite': float(sims[idx] * (1.0 + 0.5 * np.maximum(-1.0, q_z[idx])))
        })
        
    # Sort by the NEW composite score
    ranked_results = sorted(results, key=lambda x: x['composite'], reverse=True)
    
    print(f"\n=== QUALITY-WEIGHTED VAMPIRE CLUSTER (Top 100 Sim) ===")
    print(f"{'Game Name':<40} | {'Sim':<6} | {'Quality Z':<10} | {'Composite'}")
    print("-" * 75)
    for i, res in enumerate(ranked_results[:50]):
        print(f"{res['name'][:39]:<40} | {res['sim_score']:.3f} | {res['quality_z']:>10.2f} | {res['composite']:.4f}")

    print("\n--- WEED OUT TARGETS (Lowest Quality/Composite) ---")
    for i, res in enumerate(ranked_results[-10:]):
        print(f"{res['name'][:39]:<40} | {res['sim_score']:.3f} | {res['quality_z']:>10.2f} | {res['composite']:.4f}")

if __name__ == '__main__':
    audit_quality_weeding()
