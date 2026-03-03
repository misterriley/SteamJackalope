import pandas as pd
import numpy as np
import json
import ast
import time
import os

from common.constants import (
    TAG_VECTORS_FILE, TAG_NORMS_FILE, 
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA,
    TOPIC_GLOBAL_SCALING_FACTOR
)
TOPIC_MEANS_FILE = "data/production/topic_means.npy"
TOPIC_STDS_FILE = "data/production/topic_stds.npy"
from common.utils import softmin_blend

print("Loading metadata and mechanical data...")
t0 = time.time()

# Load Verbs
with open("tag_categories.json", "r") as f:
    verbs = sorted(json.load(f)["verbs"])
verb_to_idx = {v: i for i, v in enumerate(verbs)}
V = len(verbs)

# Load Diffusion Matrix & IDF
S = np.load("data/production/verb_diffusion_matrix.npy")
with open("verb_idf_weights.json", "r") as f:
    idf_dict = json.load(f)
W = np.array([idf_dict.get(v, 1.0) for v in verbs], dtype=np.float32)

# Load metadata
df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name', 'tags'])
N = len(df)

seed_game = df[df['name'] == 'Detroit: Become Human'].iloc[0]
seed_idx = df.index[df['name'] == 'Detroit: Become Human'][0]

print("Building tag matrix M...")
M = np.zeros((N, V), dtype=np.float32)
def get_active_verb_indices(tag_str):
    if not tag_str or pd.isna(tag_str) or tag_str == "None": return []
    try:
        tag_dict = ast.literal_eval(tag_str)
        if not tag_dict: return []
        max_count = max(tag_dict.values())
        return [verb_to_idx[t] for t, count in tag_dict.items() if t in verb_to_idx and (count / max_count) >= 0.1]
    except:
        return []

for i, tag_str in enumerate(df['tags']):
    indices = get_active_verb_indices(tag_str)
    if indices:
        M[i, indices] = 1.0

active_s = get_active_verb_indices(seed_game['tags'])

print("Calculating Mech Sim...")
if active_s:
    sim_to_s = np.max(S[:, active_s], axis=1)
else:
    sim_to_s = np.zeros(V)

score_a_num = M @ (sim_to_s * W)
score_a_den = M @ W
mask = score_a_den > 0
score_a = np.zeros(N, dtype=np.float32)
score_a[mask] = score_a_num[mask] / score_a_den[mask]

sims_b_num = np.zeros(N, dtype=np.float32)
sum_weights_b = 0.0
if active_s:
    for vb in active_s:
        w = W[vb]
        max_sim_b = np.max(M * S[vb, :], axis=1)
        sims_b_num += max_sim_b * w
        sum_weights_b += w

score_b = np.zeros(N, dtype=np.float32)
if sum_weights_b > 0:
    score_b = sims_b_num / sum_weights_b

mech_sim = (score_a + score_b) / 2.0

print("Loading Vibe Data...")
tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
topic_means = np.load(TOPIC_MEANS_FILE)
topic_stds = np.load(TOPIC_STDS_FILE)

print("Calculating Vibe Sims...")
# 1. Tags
seed_tag_vec = tag_vectors[seed_idx]
seed_tag_norm = tag_norms[seed_idx]
tag_sims = (np.dot(tag_vectors.astype(np.float32), seed_tag_vec.astype(np.float32)) / 
            ((tag_norms + DOT_PRODUCT_LAMBDA) * (seed_tag_norm + DOT_PRODUCT_LAMBDA))) * TAG_GLOBAL_SCALING_FACTOR

# 2. Semantics
seed_sem_vec = sem_vectors[seed_idx]
seed_sem_norm = sem_norms[seed_idx]
sem_sims_raw = (np.dot(sem_vectors.astype(np.float32), seed_sem_vec.astype(np.float32)) / 
                (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
sem_sims = sem_sims_raw / (seed_sem_norm + SEMANTIC_DOT_PRODUCT_LAMBDA)

# 3. Topics
seed_topic_dist = topic_distributions[seed_idx]
fz = (seed_topic_dist.astype(np.float32) - topic_means) / (topic_stds + 1e-9)
fz[fz < 2.5] = 0
fn = np.linalg.norm(fz) + 1e-9
fz_unit = fz / fn

topic_sims = np.zeros(N, dtype=np.float32)
batch_size = 100000 
for i in range(0, N, batch_size):
    end = min(i + batch_size, N)
    bz = (topic_distributions[i:end].astype(np.float32) - topic_means) / (topic_stds + 1e-9)
    bz[bz < 2.5] = 0
    bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
    topic_sims[i:end] = np.dot(bz / bn, fz_unit)

# Softmin with T=1.0
print("Blending Vibe and Mech...")
vibe_sim = softmin_blend([sem_sims, topic_sims * TOPIC_GLOBAL_SCALING_FACTOR], temperature=1.0)

total_sim = mech_sim * vibe_sim

df['mech_sim'] = mech_sim
df['vibe_sim'] = vibe_sim
df['total_sim'] = total_sim
df['tag_sim'] = tag_sims
df['sem_sim'] = sem_sims
df['topic_sim'] = topic_sims * TOPIC_GLOBAL_SCALING_FACTOR

results = df[df['appid'] != seed_game['appid']].copy()
results = results.sort_values(by='total_sim', ascending=False).head(20)

print(f"\nTop 20 Similar Games to Detroit: Become Human (Total Time: {time.time() - t0:.2f}s)")
header = f"{'Name':<40} | {'Total':<6} | {'Mech':<6} | {'Vibe':<6} | {'Tag':<6} | {'Sem':<6} | {'Top':<6}"
print(header)
print("-" * len(header))
for i, row in results.iterrows():
    print(f"{row['name'][:40]:<40} | {row['total_sim']:.4f} | {row['mech_sim']:.4f} | {row['vibe_sim']:.4f} | {row['tag_sim']:.4f} | {row['sem_sim']:.4f} | {row['topic_sim']:.4f}")

print("\nSpecific Lookup:")
for target in ['Heavy Rain', 'Beyond: Two Souls', 'The Walking Dead', 'Until Dawn', 'Life is Strange']:
    match = df[df['name'] == target]
    if not match.empty:
        row = match.iloc[0]
        print(f"{row['name']} -> Total: {row['total_sim']:.4f} | Mech: {row['mech_sim']:.4f} | Vibe: {row['vibe_sim']:.4f} | Tag: {row['tag_sim']:.4f} | Sem: {row['sem_sim']:.4f} | Topic: {row['topic_sim']:.4f}")
