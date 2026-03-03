import pandas as pd
import numpy as np
import ast
import json
from common.constants import METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE
from common.utils import calculate_jackalope_kernel

print("Loading data...")
df = pd.read_parquet(METADATA_FILE)
verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE)

def get_idx(appid):
    return df[df['appid'] == appid].index[0]

# Strange Horticulture (1663220) and Va-11 Hall-A (447530)
idx_s = get_idx(1663220)
idx_t = get_idx(447530)

print(f"Strange Horticulture index: {idx_s}")
print(f"Va-11 Hall-A index: {idx_t}")

# Check verb profiles
v_s = verb_profiles[idx_s]
v_t = verb_profiles[idx_t]

print(f"Witcher Profile Sum: {np.sum(v_s):.4f}")
print(f"Skyrim Profile Sum: {np.sum(v_t):.4f}")

intersection = np.minimum(v_s, v_t)
print(f"Intersection Sum: {np.sum(intersection):.4f}")

# Re-run kernel parts
tag_scaling_factor = 1.0
dot_product_lambda = 1.0
sem_lambda = 0.01

from common.constants import (
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE, TOPIC_DISTRIBUTIONS_FILE
)
sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
topic_means = np.load("data/production/topic_means.npy")
topic_stds = np.load("data/production/topic_stds.npy")

sims, components = calculate_jackalope_kernel(
    verb_profiles=verb_profiles[[idx_t]],
    seed_verb_profile=verb_profiles[idx_s],
    sem_vectors=sem_vectors[[idx_t]], sem_norms=sem_norms[[idx_t]],
    seed_sem_vec=sem_vectors[idx_s], seed_sem_norm=sem_norms[idx_s],
    topic_distributions=topic_distributions[[idx_t]], seed_topic_dist=topic_distributions[idx_s],
    topic_means=topic_means, topic_stds=topic_stds,
    tag_scaling_factor=1.0, dot_product_lambda=1.0,
    sem_scaling_factor=1.0, sem_lambda=0.01,
    return_components=True
)

print(f"Kernel Result: {sims[0]:.4f}")
print(f"Components: {components}")
