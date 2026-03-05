
import pandas as pd
import numpy as np
import os
import sys
import re
import ast

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from common.constants import (
    PRODUCTION_DATA_DIR, METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE, TOPIC_DISTRIBUTIONS_FILE,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA, SEMANTIC_GLOBAL_SCALING_FACTOR,
    SEMANTIC_DOT_PRODUCT_LAMBDA, TOPIC_GLOBAL_SCALING_FACTOR, TOPIC_DOT_PRODUCT_LAMBDA,
    SOFTMIN_TEMPERATURE
)
from common.utils import (
    calculate_jackalope_kernel, MIGS, MIG_WEIGHTS, NARRATIVE_TAGS, HORROR_MARKERS, 
    HARD_ANCHORS, extract_seed_metadata, calculate_title_hijack_mask
)

def deep_dive():
    print("Loading data artifacts...")
    metadata = pd.read_parquet(METADATA_FILE)
    verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    
    means_path = os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")
    stds_path = os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")
    topic_means = np.load(means_path).astype(np.float32) if os.path.exists(means_path) else None
    topic_stds = np.load(stds_path).astype(np.float32) if os.path.exists(stds_path) else None
    
    tone_path = os.path.join(PRODUCTION_DATA_DIR, "tone_z.npy")
    tone_z = np.load(tone_path, mmap_mode='r') if os.path.exists(tone_path) else None
    
    appid_to_idx = {int(appid): i for i, appid in enumerate(metadata['appid'])}
    
    # Pre-calculate Anchor Masks
    anchor_masks = {}
    all_anchor_tags = set()
    for tags in MIGS.values(): all_anchor_tags.update(tags)
    all_anchor_tags.update(NARRATIVE_TAGS)
    all_anchor_tags.update(HORROR_MARKERS)
    all_anchor_tags.update(HARD_ANCHORS)
    
    tag_series = metadata['tags'].fillna('').astype(str)
    for tag in all_anchor_tags:
        pattern = rf"'{re.escape(tag)}':"
        anchor_masks[tag] = tag_series.str.contains(pattern, regex=True).values

    # Target Pairs to Analyze
    pairs = [
        (1222140, 960910), # Detroit -> Heavy Rain (POSITIVE, scored 0.0000)
        (1222140, 960990), # Detroit -> Beyond Two Souls (POSITIVE, scored 0.3544 - Correct)
        (1091500, 2744390), # Cyberpunk -> Dawning Clocks (NEGATIVE, scored 0.4006 - FAIL)
    ]

    for s_id, t_id in pairs:
        s_idx, t_idx = appid_to_idx[s_id], appid_to_idx[t_id]
        s_name, t_name = metadata.iloc[s_idx]['name'], metadata.iloc[t_idx]['name']
        
        print(f"\n================================================================================")
        print(f"PAIR: {s_name} (Seed) -> {t_name} (Target)")
        print(f"================================================================================")
        
        # 1. Compare MIGs
        s_meta = extract_seed_metadata([s_idx], metadata)
        t_meta = extract_seed_metadata([t_idx], metadata)
        
        s_migs = s_meta['migs_list'][0]
        t_migs = t_meta['migs_list'][0]
        
        print(f"Seed MIGs: {s_migs}")
        print(f"Target MIGs: {t_migs}")
        
        common_migs = s_migs.intersection(t_migs)
        print(f"Shared MIGs: {common_migs}")
        
        # 2. Manual Identity Match Calculation (Weighted)
        intersection_w = 0.0
        union_w = 0.0
        for group in MIGS.keys():
            w = MIG_WEIGHTS.get(group, 1.0)
            s_has = group in s_migs
            t_has = group in t_migs
            
            if s_has and t_has:
                intersection_w += w
                union_w += w
            elif s_has or t_has:
                union_w += w
        
        id_match = intersection_w / (union_w + 1e-9)
        print(f"Manual Weighted ID Match: {id_match:.4f} (ID^3: {id_match**3:.4f})")
        
        # 3. Component Breakdown from Kernel
        scores, components = calculate_jackalope_kernel(
            verb_profiles=verb_profiles, seed_verb_profile=verb_profiles[s_idx],
            sem_vectors=sem_vectors, sem_norms=sem_norms, seed_sem_vec=sem_vectors[s_idx], seed_sem_norm=sem_norms[s_idx],
            topic_distributions=topic_distributions, seed_topic_dist=topic_distributions[s_idx],
            topic_means=topic_means, topic_stds=topic_stds,
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
            sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
            seed_migs=s_migs, seed_tags=s_meta['all_soul_tags'],
            candidate_anchor_masks=anchor_masks,
            difficulty_z=metadata['difficulty_z'].values, seed_difficulty_z=metadata.iloc[s_idx]['difficulty_z'],
            tone_z=tone_z if tone_z is not None else metadata['tone_z'].values,
            seed_tone_z=tone_z[s_idx] if tone_z is not None else metadata.iloc[s_idx]['tone_z'],
            return_components=True
        )
        
        c = {k: float(v[t_idx]) for k, v in components.items()}
        print(f"Kernel Score: {scores[t_idx]:.4f}")
        print(f"Components: {c}")
        
        # 4. Check for Hard Vetoes (Perspective, NSFW, VR)
        # We'll simulate apply_kernel_vetoes logic manually
        is_3d_s = any(p in s_meta['all_soul_tags'] for p in ["3D", "Third Person", "First-Person"])
        is_3d_t = any(p in t_meta['all_soul_tags'] for p in ["3D", "Third Person", "First-Person"])
        print(f"Seed is 3D: {is_3d_s} | Target is 3D: {is_3d_t}")
        
        # 5. Check Tag Sets for "The Dawning Clocks" failure
        if t_id == 2744390:
            print(f"\nAnalyzing 'The Dawning Clocks of Time' Failure:")
            print(f"Tags: {metadata.iloc[t_idx]['tags']}")

if __name__ == "__main__":
    deep_dive()
