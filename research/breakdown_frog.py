import pandas as pd
import numpy as np
import ast
import re
from common.utils import calculate_jackalope_kernel, MIGS, NARRATIVE_TAGS, HORROR_MARKERS, HARD_ANCHORS, softmin_blend
from common.constants import (
    METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA,
    EPSILON
)

def breakdown():
    df = pd.read_parquet(METADATA_FILE)
    verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    topic_means = np.load("data/production/topic_means.npy").astype(np.float32)
    topic_stds = np.load("data/production/topic_stds.npy").astype(np.float32)

    seed_appid = 1194840 # Frog Fractions
    target_appid = 1379510 # Algebra Ridge
    
    idx_s = df[df['appid'] == seed_appid].index[0]
    idx_t = df[df['appid'] == target_appid].index[0]
    
    # 1. Prepare Metadata
    tags_s_dict = ast.literal_eval(df.iloc[idx_s]['tags'])
    max_s = max(tags_s_dict.values()) if tags_s_dict else 1.0
    seed_tags_strict = {t for t, v in tags_s_dict.items() if v / max_s > 0.35}
    seed_tags_soul = {t for t, v in tags_s_dict.items() if v / max_s > 0.15}
    
    seed_migs = {group for group, tags in MIGS.items() if any(t in seed_tags_strict for t in tags)}
    active_narrative = [t for t in NARRATIVE_TAGS if t in seed_tags_soul]
    
    # Pre-calculate masks for target
    all_anchor_tags = set()
    for tags in MIGS.values(): all_anchor_tags.update(tags)
    all_anchor_tags.update(NARRATIVE_TAGS)
    all_anchor_tags.update(HORROR_MARKERS)
    all_anchor_tags.update(HARD_ANCHORS)
    all_anchor_tags.update({"Education", "Math", "Science", "Typing", "Spelling", "Programming", "Logic"})
    all_anchor_tags.update({"Surreal", "Comedy", "Funny", "Satire", "Parody", "Memes", "Abstract"})
    
    target_tags_dict = ast.literal_eval(df.iloc[idx_t]['tags'])
    candidate_masks = {t: np.array([t in target_tags_dict]) for t in all_anchor_tags}

    total_sim, components = calculate_jackalope_kernel(
        verb_profiles=verb_profiles[[idx_t]],
        seed_verb_profile=verb_profiles[idx_s],
        sem_vectors=sem_vectors[[idx_t]], sem_norms=sem_norms[[idx_t]],
        seed_sem_vec=sem_vectors[idx_s], seed_sem_norm=sem_norms[idx_s],
        topic_distributions=topic_distributions[[idx_t]], seed_topic_dist=topic_distributions[idx_s],
        topic_means=topic_means, topic_stds=topic_stds,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
        seed_migs=seed_migs,
        seed_tags=seed_tags_soul,
        candidate_anchor_masks=candidate_masks,
        active_narrative_seed=active_narrative,
        return_components=True
    )
    
    print(f"Seed: {df.iloc[idx_s]['name']}")
    print(f"Target: {df.iloc[idx_t]['name']}")
    print(f"Final Sim (Kernel): {total_sim[0]:.6f}")
    
    # Vibe Sim Calculation (Manual CDF replacement for diagnostic)
    # Since rank(pct) is always 1.0 for single candidate, 
    # the 0.66 score must be coming from something else.
    
    print(f"Components: {components}")

if __name__ == "__main__":
    breakdown()
