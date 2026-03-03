import json
import numpy as np
import os
import pandas as pd
import re

def softmin_blend(signals, temperature=0.01):
    stack = np.stack(signals, axis=0)
    scaled = -stack / temperature
    max_val = np.max(scaled, axis=0)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals, axis=0)
    return np.sum(stack * weights, axis=0)

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))
t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))

with open(os.path.join(PRODUCTION_DATA_DIR, 'motivations_library.json'), 'r') as f:
    library = json.load(f)

# Focus on Completion
vectors = library['Completion']
m_tag_names = vectors.get('tags', [])
m_tag_pattern = "|".join([re.escape(t) for t in m_tag_names])
all_tag_sims_m = df['tags'].fillna('').astype(str).str.contains(m_tag_pattern, regex=True).astype(np.float32) * 0.2

# We need the max consensus_sim for Completion
# I won't run the full semantic/topic sim here, but I'll check the tags first.
print(f"Max Tag Sim for Completion: {np.max(all_tag_sims_m)}")
print(f"Number of games with Completion tags: {np.sum(all_tag_sims_m > 0)}")

# Let's check a specific game that SHOULD match Completion
# e.g. a game with lots of achievements or collectables
# Completion tags: Achievement, Collectathon, 100%, etc.
print(f"Completion tags: {m_tag_names}")
