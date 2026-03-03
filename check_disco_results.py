import pandas as pd
import numpy as np
import ast
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.constants import *
from common.utils import calculate_jackalope_kernel, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS

def analyze_disco_mismatch():
    df = pd.read_parquet(METADATA_FILE)
    aid_to_idx = {int(a): i for i, a in enumerate(df['appid'])}
    
    seed_appid = 632470 # Disco Elysium
    bad_appids = [4273730, 4273700, 1965190]
    
    s_idx = aid_to_idx[seed_appid]
    
    # Load all needed vectors
    sem_v = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_n = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    verb_p = np.load(os.path.join(PRODUCTION_DATA_DIR, 'diffused_verb_profiles.npy'), mmap_mode='r')
    top_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy')).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy')).astype(np.float32)
    
    # Pre-calculate anchor masks
    all_needed_tags = set()
    for tags in MIGS.values(): all_needed_tags.update(tags)
    all_needed_tags.update(NARRATIVE_TAGS)
    all_needed_tags.update(HORROR_MARKERS)
    all_needed_tags.update(HARD_ANCHORS)
    all_needed_tags.add("Isometric")
    all_needed_tags.add("CRPG")
    
    # Extract Seed Metadata
    tags_s_raw = df.iloc[s_idx]['tags']
    tags_s = ast.literal_eval(tags_s_raw) if isinstance(tags_s_raw, str) else tags_s_raw
    max_s = max(tags_s.values()) if tags_s else 1.0
    seed_tags_strict = {t for t, v in tags_s.items() if v / max_s > 0.25}
    seed_tags_soul = {t for t, v in tags_s.items() if v / max_s > 0.15}
    seed_migs = {group for group, tags in MIGS.items() if any(t in seed_tags_strict for t in tags)}
    active_narrative = [t for t in NARRATIVE_TAGS if t in seed_tags_soul]
    
    print(f"Seed: {df.iloc[s_idx]['name']} | Seed MIGs: {seed_migs}\n")
    print(f"Seed Tags (Strict): {seed_tags_strict}\n")

    # Background population for CDF
    bg_indices = np.random.choice(len(df), size=1000, replace=False)

    for appid in bad_appids:
        target_idx = aid_to_idx[appid]
        test_indices = np.append(bg_indices, target_idx)
        
        sim, comps = calculate_jackalope_kernel(
            verb_profiles=verb_p[test_indices], seed_verb_profile=verb_p[s_idx],
            sem_vectors=sem_v[test_indices], sem_norms=sem_n[test_indices],
            seed_sem_vec=sem_v[s_idx], seed_sem_norm=sem_n[s_idx],
            topic_distributions=top_dist[test_indices], seed_topic_dist=top_dist[s_idx],
            topic_means=t_means, topic_stds=t_stds,
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
            sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
            mature_content_flags=np.array([df.iloc[i]['mature_content'] > 0 for i in test_indices]),
            seed_mature_content=bool(df.iloc[s_idx]['mature_content'] > 0),
            seed_migs=seed_migs,
            seed_tags=seed_tags_strict,
            candidate_anchor_masks={t: np.array([t in (ast.literal_eval(df.iloc[i]['tags']) if isinstance(df.iloc[i]['tags'], str) else df.iloc[i]['tags']) for i in test_indices]) for t in all_needed_tags},
            active_narrative_seed=active_narrative,
            difficulty_z=df.iloc[test_indices]['difficulty_z'].values,
            seed_difficulty_z=df.iloc[s_idx]['difficulty_z'],
            tone_z=df.iloc[test_indices]['tone_z'].values,
            seed_tone_z=df.iloc[s_idx]['tone_z'],
            return_components=True
        )
        
        def get_val(key):
            val = comps[key]
            return val[-1] if hasattr(val, '__getitem__') else val

        print(f"Target: {df.iloc[target_idx]['name']} | Total: {float(sim[-1]):.4f}")
        print(f"  - Vibe(Tag): {float(get_val('vibe')):.4f} | Theme(Sem): {float(get_val('theme')):.4f} | Cluster(Topic): {float(get_val('cluster')):.4f}")
        print(f"  - Combined: {float(get_val('combined')):.4f} | ToneS: {float(get_val('tone')):.4f} | DiffS: {float(get_val('difficulty')):.4f}")

if __name__ == '__main__':
    analyze_disco_mismatch()
