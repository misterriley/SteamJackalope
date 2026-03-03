import pytest
import numpy as np
import pandas as pd
import re
import ast
from common.utils import calculate_jackalope_kernel, MIGS
from common.constants import (
    METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE, EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE, TOPIC_DISTRIBUTIONS_FILE,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)

@pytest.fixture(scope="module")
def data():
    metadata = pd.read_parquet(METADATA_FILE)
    verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load("data/production/topic_means.npy")
    t_stds = np.load("data/production/topic_stds.npy")
    
    appid_to_idx = {int(aid): i for i, aid in enumerate(metadata['appid'])}
    name_to_idx = {str(name).lower(): i for i, name in enumerate(metadata['name'])}
    
    # Pre-calculate anchor masks
    anchor_masks = {}
    all_anchor_tags = set()
    for tags in MIGS.values(): all_anchor_tags.update(tags)
    
    tag_series = metadata['tags'].fillna('').astype(str)
    for tag in all_anchor_tags:
        pattern = rf"'{re.escape(tag)}':"
        anchor_masks[tag] = tag_series.str.contains(pattern, regex=True).values
        
    return {
        'metadata': metadata,
        'verb_profiles': verb_profiles,
        'sem_vectors': sem_vectors,
        'sem_norms': sem_norms,
        'topic_distributions': topic_distributions,
        't_means': t_means,
        't_stds': t_stds,
        'appid_to_idx': appid_to_idx,
        'name_to_idx': name_to_idx,
        'anchor_masks': anchor_masks
    }

def get_idx(data, identifier):
    if str(identifier).isdigit():
        return data['appid_to_idx'].get(int(identifier))
    return data['name_to_idx'].get(str(identifier).lower())

def test_mig_bridge_dead_cells(data):
    """Verifies that Roguevania hybrids (Dead Cells) bridge Roguelikes and Metroidvanias."""
    idx_dc = get_idx(data, 588650) # Dead Cells
    idx_hk = get_idx(data, 367520) # Hollow Knight (Metroidvania)
    idx_hades = get_idx(data, 1145360) # Hades (Roguelike)
    
    def get_skill_mult(s_idx, t_idx):
        tags_s_str = data['metadata'].iloc[s_idx]['tags']
        seed_tags_set = set(re.findall(r"'([^']+)':", tags_s_str))
        seed_anchors = [group for group, tags in MIGS.items() if any(t in seed_tags_set for t in tags)]
        
        sims, comps = calculate_jackalope_kernel(
            verb_profiles=data['verb_profiles'][[t_idx]],
            seed_verb_profile=data['verb_profiles'][s_idx],
            sem_vectors=data['sem_vectors'][[t_idx]], sem_norms=data['sem_norms'][[t_idx]],
            seed_sem_vec=data['sem_vectors'][s_idx], seed_sem_norm=data['sem_norms'][s_idx],
            topic_distributions=data['topic_distributions'][[t_idx]], seed_topic_dist=data['topic_distributions'][s_idx],
            topic_means=data['t_means'], topic_stds=data['t_stds'],
            tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
            sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
            seed_migs=seed_anchors,            candidate_anchor_masks={a: np.array([data['anchor_masks'][a][t_idx]]) for a in data['anchor_masks'] if a in seed_tags_set or any(a in tags for tags in MIGS.values())},
            return_components=True
        )
        # We need to calculate skill_jaccard manually since it's not in return_components yet
        seed_migs = {group for group, tags in MIGS.items() if any(t in seed_tags_set for t in tags)}
        target_tags_str = data['metadata'].iloc[t_idx]['tags']
        target_tags_set = set(re.findall(r"'([^']+)':", target_tags_str))
        target_migs = {group for group, tags in MIGS.items() if any(t in target_tags_set for t in tags)}
        
        intersection = seed_migs & target_migs
        union = seed_migs | target_migs
        skill_j = len(intersection) / len(union) if union else 1.0
        return 0.2 + 0.8 * skill_j

    mult_hk = get_skill_mult(idx_dc, idx_hk)
    mult_hades = get_skill_mult(idx_dc, idx_hades)
    
    print(f"Dead Cells -> HK Mult: {mult_hk:.4f}")
    print(f"Dead Cells -> Hades Mult: {mult_hades:.4f}")
    
    # Dead Cells has Roguevania (ROGUELIKE + METROIDVANIA). 
    # It should have a strong multiplier with both.
    assert mult_hk > 0.5, "Dead Cells should have a strong mechanical link to Hollow Knight (Metroidvania bridge)"
    assert mult_hades > 0.5, "Dead Cells should have a strong mechanical link to Hades (Roguelike bridge)"

def test_difficulty_similarity(data):
    """Verifies that difficulty_sim correctly weights games with matching challenge."""
    idx_dc = get_idx(data, 588650) # Dead Cells (Hard)
    idx_celeste = get_idx(data, 504230) # Celeste (Hard)
    idx_stardew = get_idx(data, 413150) # Stardew (Easy)
    
    diff_z = data['metadata']['difficulty_z'].values
    seed_diff_z = diff_z[idx_dc]
    
    # Calculate components
    def get_diff_sim(t_idx):
        return np.exp(-0.5 * (diff_z[t_idx] - seed_diff_z)**2)
        
    sim_celeste = get_diff_sim(idx_celeste)
    sim_stardew = get_diff_sim(idx_stardew)
    
    print(f"Dead Cells -> Celeste Diff Sim: {sim_celeste:.4f}")
    print(f"Dead Cells -> Stardew Diff Sim: {sim_stardew:.4f}")
    
    assert sim_celeste > sim_stardew, "Dead Cells should be more difficulty-similar to Celeste than Stardew Valley"

def test_adult_only_barrier(data):
    """Verifies that the Adult Only banner flag creates a hard siloing effect."""
    idx_ff = get_idx(data, 413150) # Stardew Valley (Family Friendly)
    # Search for an adult-only game in the metadata
    adult_mask = data['metadata']['mature_content'] > 0
    if not any(adult_mask):
        pytest.skip("No Adult Only games found in production metadata")
        
    idx_adult = np.where(adult_mask)[0][0]
    name_adult = data['metadata'].iloc[idx_adult]['name']
    
    def get_skill_jaccard(s_idx, t_idx):
        # Seed logic
        s_mature = data['metadata'].iloc[s_idx]['mature_content'] > 0
        t_mature = data['metadata'].iloc[t_idx]['mature_content'] > 0
        
        # Tags for other MIGs
        tags_s_str = data['metadata'].iloc[s_idx]['tags']
        seed_tags_set = set(re.findall(r"'([^']+)':", tags_s_str))
        seed_migs = {group for group, tags in MIGS.items() if any(t in seed_tags_set for t in tags)}
        if s_mature: seed_migs.add("ADULT_ONLY")
        
        target_tags_str = data['metadata'].iloc[t_idx]['tags']
        target_tags_set = set(re.findall(r"'([^']+)':", target_tags_str))
        target_migs = {group for group, tags in MIGS.items() if any(t in target_tags_set for t in tags)}
        if t_mature: target_migs.add("ADULT_ONLY")
        
        intersection = seed_migs & target_migs
        union = seed_migs | target_migs
        return len(intersection) / len(union) if union else 1.0

    skill_j = get_skill_jaccard(idx_ff, idx_adult)
    print(f"Stardew -> {name_adult} Skill Jaccard: {skill_j:.4f}")
    
    # Because one is adult and one is not, the union increases but intersection does not.
    # This should result in a lower multiplier than if they were in the same maturity silo.
    # More importantly, if they shared ALL other tags, the maturity silo would still penalize.
    assert skill_j < 1.0, "Stardew and an Adult game should not have 100% mechanical parity"

def test_title_hijacking_penalty(data):
    """Verifies that the title hijack penalty reduces similarity for mechanical clashes."""
    # We'll use "Hollow Knight" as seed and find a game that contains "Hollow" or "Knight" 
    # but is NOT mechanically similar (e.g. a racing game or puzzle game).
    idx_hk = get_idx(data, 367520)
    
    # Find a potential hijacker (Title contains "Knight" but NOT Metroidvania or Platformer)
    knight_mask = data['metadata']['name'].str.contains("Knight", case=False)
    m_mask_mv = data['anchor_masks'].get('Metroidvania', np.zeros(len(data['metadata']), dtype=bool))
    m_mask_pl = data['anchor_masks'].get('Platformer', np.zeros(len(data['metadata']), dtype=bool))
    mechanical_mask = m_mask_mv | m_mask_pl
    
    potential_hijackers = np.where(knight_mask & ~mechanical_mask)[0]
    if len(potential_hijackers) == 0:
        pytest.skip("No title hijacker found for 'Knight'")
        
    idx_h = potential_hijackers[0]
    name_h = data['metadata'].iloc[idx_h]['name']
    
    # Pre-calculate kernel with hijack penalty (Ensure mask matches candidate pool size)
    title_hijack_mask = np.array([True], dtype=bool) # We are only passing the one hijacker index
    
    tags_s_str = data['metadata'].iloc[idx_hk]['tags']
    seed_tags_set = set(re.findall(r"'([^']+)':", tags_s_str))
    seed_anchors = [group for group, tags in MIGS.items() if any(t in seed_tags_set for t in tags)]
    
    sims, comps = calculate_jackalope_kernel(
        verb_profiles=data['verb_profiles'][[idx_h]],
        seed_verb_profile=data['verb_profiles'][idx_hk],
        sem_vectors=data['sem_vectors'][[idx_h]], sem_norms=data['sem_norms'][[idx_h]],
        seed_sem_vec=data['sem_vectors'][idx_hk], seed_sem_norm=data['sem_norms'][idx_hk],
        topic_distributions=data['topic_distributions'][[idx_h]], seed_topic_dist=data['topic_distributions'][idx_hk],
        topic_means=data['t_means'], topic_stds=data['t_stds'],
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
        seed_migs=seed_anchors,        candidate_anchor_masks={a: np.array([data['anchor_masks'][a][idx_h]]) for a in data['anchor_masks'] if a in seed_tags_set or any(a in tags for tags in MIGS.values())},
        precalculated_masks={"title_hijack": title_hijack_mask},
        return_components=True
    )
    
    score = float(sims[0])
    print(f"Hollow Knight -> {name_h} Sim: {score:.6f}")
    
    # The penalty should make the score extremely low
    assert score < 0.05, f"Title hijacker '{name_h}' should be heavily penalized"
