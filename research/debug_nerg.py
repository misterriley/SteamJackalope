import pandas as pd
import numpy as np
import ast
import json
import os
from common.utils import calculate_jackalope_kernel
from common.constants import (
    METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR
)

df = pd.read_parquet(METADATA_FILE)
verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
topic_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
topic_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

seed_appid = 1194840 # Frog Fractions
target_appid = 1370960 # Nerg Loves Them

s_idx = df[df['appid'] == seed_appid].index[0]
t_idx = df[df['appid'] == target_appid].index[0]

s_row = df.iloc[s_idx]
t_row = df.iloc[t_idx]

print(f"Seed: {s_row['name']} ({seed_appid})")
print(f"Target: {t_row['name']} ({target_appid})")
print(f"Target Desc: {t_row['short_description']}")
print(f"Target Tags: {t_row['tags']}")

sim, comps = calculate_jackalope_kernel(
    verb_profiles=verb_profiles[[t_idx]],
    seed_verb_profile=verb_profiles[s_idx],
    sem_vectors=sem_vectors[[t_idx]], sem_norms=sem_norms[[t_idx]],
    seed_sem_vec=sem_vectors[s_idx], seed_sem_norm=sem_norms[s_idx],
    topic_distributions=topic_distributions[[t_idx]], seed_topic_dist=topic_distributions[s_idx],
    topic_means=topic_means, topic_stds=topic_stds,
    tag_scaling_factor=1.0, dot_product_lambda=1.0,
    sem_scaling_factor=1.0, sem_lambda=0.01,
    return_components=True
)

print(f"\nSimilarity Components: {comps}")
print(f"Raw Score: {sim[0]:.4f}")

# Check Rescue Nouns
tags_s = s_row['tags']
if isinstance(tags_s, str): tags_s = ast.literal_eval(tags_s)
tags_t = t_row['tags']
if isinstance(tags_t, str): tags_t = ast.literal_eval(tags_t)

HIGH_VALUE_NOUNS = {'Education', 'Math', 'Comedy', 'Surreal', 'Typing', 'Spelling', 'Mystery', 'Word Game'}
shared = set(tags_s.keys()) & set(tags_t.keys()) & HIGH_VALUE_NOUNS
print(f"Shared HV Nouns: {shared}")

# Verb analysis
with open("tag_categories.json", "r") as f:
    verbs = sorted(json.load(f)["verbs"])

def get_top_verbs(idx):
    prof = verb_profiles[idx]
    active = [(verbs[i], prof[i]) for i in range(len(verbs)) if prof[i] > 0.5]
    return sorted(active, key=lambda x: x[1], reverse=True)[:5]

print(f"Seed Verbs: {get_top_verbs(s_idx)}")
print(f"Target Verbs: {get_top_verbs(t_idx)}")
