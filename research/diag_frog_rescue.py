import pandas as pd
import numpy as np
import ast
import json
import re
import os
from common.utils import calculate_jackalope_kernel
from common.constants import (
    METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)

df = pd.read_parquet(METADATA_FILE)
verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
topic_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
topic_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

seed_appid = 1194840 # Frog Fractions
target_appid = 1379510 # Algebra Ridge

match = df[df['appid'] == seed_appid]
seed_idx = match.index[0]
tags_s = match.iloc[0]['tags']
if isinstance(tags_s, str): tags_s = ast.literal_eval(tags_s)

def diag(target_id):
    t_match = df[df['appid'] == target_id]
    t_idx = t_match.index[0]
    t_name = t_match.iloc[0]['name']
    t_tags = t_match.iloc[0]['tags']
    if isinstance(t_tags, str): t_tags = ast.literal_eval(t_tags)
    
    # Calculate Kernel
    sim = calculate_jackalope_kernel(
        verb_profiles=verb_profiles[[t_idx]],
        seed_verb_profile=verb_profiles[seed_idx],
        sem_vectors=sem_vectors[[t_idx]], sem_norms=sem_norms[[t_idx]],
        seed_sem_vec=sem_vectors[seed_idx], seed_sem_norm=sem_norms[seed_idx],
        topic_distributions=topic_distributions[[t_idx]], seed_topic_dist=topic_distributions[seed_idx],
        topic_means=topic_means, topic_stds=topic_stds,
        tag_scaling_factor=1.0, dot_product_lambda=1.0,
        sem_scaling_factor=1.0, sem_lambda=0.01
    )
    
    # Check Rescue Nouns
    HIGH_VALUE_NOUNS = {'Education', 'Math', 'Comedy', 'Surreal', 'Typing', 'Spelling', 'Mystery', 'Word Game'}
    shared = set(tags_s.keys()) & set(t_tags.keys()) & HIGH_VALUE_NOUNS
    
    print(f"\nGame: {t_name}")
    print(f"  Kernel Sim: {sim[0]:.4f}")
    print(f"  Shared HV Nouns: {shared}")
    print(f"  Final Boosted Score: {sim[0] + (0.15 if shared else 0.0):.4f}")

diag(1379510) # Algebra Ridge
diag(332570)  # Amazing Frog?
diag(2998670) # Flying Frogs
