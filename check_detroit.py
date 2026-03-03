import pandas as pd
import numpy as np
import ast
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.constants import *
from common.utils import calculate_jackalope_kernel, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS

def analyze_detroit():
    df = pd.read_parquet(METADATA_FILE)
    aid_to_idx = {int(a): i for i, a in enumerate(df['appid'])}
    
    detroit_appid = 1222140
    bad_appids = [1371300, 1018960, 2726370] # Cuba2077, Thugs Law, Suck Up!
    
    s_idx = aid_to_idx[detroit_appid]
    
    # Load all needed vectors
    sem_v = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_n = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    verb_p = np.load(os.path.join(PRODUCTION_DATA_DIR, 'diffused_verb_profiles.npy'), mmap_mode='r')
    top_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy')).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy')).astype(np.float32)
    
    # Pre-calculate anchor masks for Candidates
    all_needed_tags = set()
    for tags in MIGS.values(): all_needed_tags.update(tags)
    all_needed_tags.update(NARRATIVE_TAGS)
    all_needed_tags.update(HORROR_MARKERS)
    all_needed_tags.update(HARD_ANCHORS)
    
    # Extract Seed Metadata
    tags_s_raw = df.iloc[s_idx]['tags']
    tags_s = ast.literal_eval(tags_s_raw) if isinstance(tags_s_raw, str) else tags_s_raw
    max_s = max(tags_s.values()) if tags_s else 1.0
    seed_tags_strict = {t for t, v in tags_s.items() if v / max_s > 0.35}
    seed_tags_soul = {t for t, v in tags_s.items() if v / max_s > 0.15}
    seed_migs = {group for group, tags in MIGS.items() if any(t in seed_tags_strict for t in tags)}
    active_narrative = [t for t in NARRATIVE_TAGS if t in seed_tags_soul]
    is_cinematic_s = "Cinematic" in seed_tags_soul
    
    print(f"Seed: {df.iloc[s_idx]['name']} (Cinematic: {is_cinematic_s})\n")

    for bad_id in bad_appids:
        target_idx = aid_to_idx[bad_id]
        cand_tags_raw = df.iloc[target_idx]['tags']
        cand_tags_dict = ast.literal_eval(cand_tags_raw) if isinstance(cand_tags_raw, str) else cand_tags_raw
        cand_masks = {t: np.array([t in cand_tags_dict]) for t in all_needed_tags}
        
        sim = calculate_jackalope_kernel(
            verb_profiles=verb_p[[target_idx]], seed_verb_profile=verb_p[s_idx],
            sem_vectors=sem_v[[target_idx]], sem_norms=sem_n[[target_idx]],
            seed_sem_vec=sem_v[s_idx], seed_sem_norm=sem_n[s_idx],
            topic_distributions=top_dist[[target_idx]], seed_topic_dist=top_dist[s_idx],
            topic_means=t_means, topic_stds=t_stds,
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
            sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
            mature_content_flags=np.array([df.iloc[target_idx]['mature_content'] > 0]),
            seed_mature_content=bool(df.iloc[s_idx]['mature_content'] > 0),
            seed_migs=seed_migs,
            seed_tags=seed_tags_strict,
            candidate_anchor_masks=cand_masks,
            active_narrative_seed=active_narrative,
            is_cinematic_seed=is_cinematic_s,
            difficulty_z=df.iloc[[target_idx]]['difficulty_z'].values,
            seed_difficulty_z=df.iloc[s_idx]['difficulty_z'],
            tone_z=df.iloc[[target_idx]]['tone_z'].values,
            seed_tone_z=df.iloc[s_idx]['tone_z']
        )
        
        print(f"Target: {df.iloc[target_idx]['name']} | Sim: {float(sim[0]):.4f}")

if __name__ == '__main__':
    analyze_detroit()
