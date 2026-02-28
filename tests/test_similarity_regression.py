import pytest
import pandas as pd
import numpy as np
import os
import sys
import json
import ast
import re

# Add parent directory to sys.path for test imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, PRODUCTION_DATA_DIR, TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA, TAG_GLOBAL_SCALING_FACTOR, EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE, SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, TOPIC_DISTRIBUTIONS_FILE
)
from common.utils import calculate_jackalope_kernel

@pytest.fixture(scope="module")
def data():
    """Load production data once for all similarity tests."""
    full_metadata = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): i for i, aid in enumerate(full_metadata['appid'])}
    name_to_idx = {str(name).lower(): i for i, name in enumerate(full_metadata['name'])}
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

    STRICT_ANCHORS = ["Platformer", "Puzzle", "Strategy", "RPG", "Roguelike", "Souls-like", "Metroidvania", "JRPG", "Survival", "Visual Novel", "Anime", "FPS", "Third Person", "Shooter", "Turn-Based Combat", "Turn-Based Strategy", "Turn-Based Tactics", "Hack and Slash", "Spectacle fighter"]
    HORROR_MARKERS = ["Horror", "Survival Horror", "Psychological Horror", "Gore", "Violent"]
    PLATFORMER_VARIANTS = ["2D Platformer", "3D Platformer", "Precision Platformer", "Puzzle Platformer"]
    MISC_VARIANTS = ["First-Person", "TPS", "Looter Shooter", "Action-Adventure", "Adventure"]
    ALL_ANCHORS = STRICT_ANCHORS + HORROR_MARKERS + PLATFORMER_VARIANTS + MISC_VARIANTS
    
    anchor_masks = {a: full_metadata['tags'].fillna('').astype(str).str.contains(rf"'{re.escape(a)}':", regex=True).values for a in ALL_ANCHORS}

    return {
        'metadata': full_metadata,
        'appid_to_idx': appid_to_idx,
        'name_to_idx': name_to_idx,
        'tag_vectors': tag_vectors,
        'tag_norms': tag_norms,
        'sem_vectors': sem_vectors,
        'sem_norms': sem_norms,
        'topic_distributions': topic_distributions,
        't_means': t_means,
        't_stds': t_stds,
        'anchor_masks': anchor_masks,
        'STRICT_ANCHORS': STRICT_ANCHORS,
        'HORROR_MARKERS': HORROR_MARKERS,
        'ALL_ANCHORS': ALL_ANCHORS
    }

def get_regression_examples():
    """Load examples from the JSON file."""
    path = os.path.join("tests", "regression_tests.json")
    with open(path, "r", encoding="utf-8") as f:
        suite = json.load(f)
    
    params = []
    # Positives
    for ex in suite.get('positive_examples', []):
        if "Seed:" in ex['context']:
            seed = ex['context'].replace("Seed: ", "").strip()
            params.append((seed, ex['target'], True, ex['reason']))
    
    # Negatives
    for ex in suite.get('negative_examples', []):
        if "Seed:" in ex['context']:
            seed = ex['context'].replace("Seed: ", "").strip()
            params.append((seed, ex['target'], False, ex['reason']))
            
    return params

@pytest.mark.parametrize("seed_name, target_name, should_be_similar, reason", get_regression_examples())
def test_similarity_regression(data, seed_name, target_name, should_be_similar, reason):
    """Verifies that specific game pairs maintain their expected similarity status."""
    
    def get_idx(identifier):
        if str(identifier).isdigit():
            return data['appid_to_idx'].get(int(identifier))
        return data['name_to_idx'].get(str(identifier).lower())

    idx_s = get_idx(seed_name)
    idx_t = get_idx(target_name)
    
    assert idx_s is not None, f"Seed game '{seed_name}' not found in metadata."
    assert idx_t is not None, f"Target game '{target_name}' not found in metadata."
    
    full_metadata = data['metadata']
    
        # Calculate Kernel Similarity
        tags_s = full_metadata.iloc[idx_s]['tags']
        if isinstance(tags_s, str): tags_s = ast.literal_eval(tags_s)
        max_s = max(tags_s.values()) if tags_s else 1.0
        seed_anchors = [a for a in data['STRICT_ANCHORS'] if tags_s.get(a, 0) / max_s > 0.25]
        
        # Narrative Context
    
    NARRATIVE_TAGS = ["Story Rich", "Choices Matter", "Multiple Endings", "Visual Novel", "Atmospheric", "Emotional"]
    active_narrative = [t for t in NARRATIVE_TAGS if tags_s.get(t, 0) / max_s > 0.3]

    sims = calculate_jackalope_kernel(
        tag_vectors=data['tag_vectors'][[idx_t]], tag_norms=data['tag_norms'][[idx_t]],
        seed_tag_vec=data['tag_vectors'][idx_s], seed_tag_norm=data['tag_norms'][idx_s],
        sem_vectors=data['sem_vectors'][[idx_t]], sem_norms=data['sem_norms'][[idx_t]],
        seed_sem_vec=data['sem_vectors'][idx_s], seed_sem_norm=data['sem_norms'][idx_s],
        topic_distributions=data['topic_distributions'][[idx_t]], seed_topic_dist=data['topic_distributions'][idx_s],
        topic_means=data['t_means'], topic_stds=data['t_stds'],
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA,
        seed_anchors=seed_anchors,
        active_narrative_seed=active_narrative,
        candidate_anchor_masks={a: np.array([data['anchor_masks'][a][idx_t]]) for a in data['ALL_ANCHORS']}
    )
    
    score = float(sims[0])
    
    if should_be_similar:
        assert score > 0.05, f"FAILED POSITIVE: {seed_name} -> {target_name} | Sim: {score:.4f} | Reason: {reason}"
    else:
        assert score < 0.01, f"FAILED NEGATIVE: {seed_name} -> {target_name} | Sim: {score:.4f} | Reason: {reason}"
