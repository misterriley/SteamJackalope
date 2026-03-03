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
    TOPIC_GLOBAL_SCALING_FACTOR,
    SEMANTIC_SIMILARITY_MEAN, SEMANTIC_SIMILARITY_STD,
    TOPIC_SIMILARITY_MEAN, TOPIC_SIMILARITY_STD
)
from scipy.stats import norm
TOPIC_MEANS_FILE = "data/production/topic_means.npy"
TOPIC_STDS_FILE = "data/production/topic_stds.npy"
from common.utils import softmin_blend

print("Loading metadata and mechanical data...")
t0 = time.time()

with open("tag_categories.json", "r") as f:
    verbs = sorted(json.load(f)["verbs"])
verb_to_idx = {v: i for i, v in enumerate(verbs)}
V = len(verbs)

S = np.load("data/production/verb_diffusion_matrix.npy")
with open("verb_idf_weights.json", "r") as f:
    idf_dict = json.load(f)

implicit_verbs = {
    "Exploration", "Conversation", "Diplomacy", "Trading", "Gambling", "Destruction",
    "Automation", "Multiple Endings", "Collectathon", "Narration", "Audio Production",
    "Video Production", "Photo Editing", 
    "Cinematic", "Emotional", "Horror", "Psychological Horror", "Story Rich", "Relaxing",
    "Competitive", "Violent", "Choices Matter", "Comedy"
}

W_list = []
for v in verbs:
    w = idf_dict.get(v, 1.0)
    if v in implicit_verbs:
        w *= 0.5
    W_list.append(w)
W = np.array(W_list, dtype=np.float32)

df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name', 'tags'])
N = len(df)

seed_name = "A Plague Tale: Innocence"
seed_game = df[df['name'] == seed_name].iloc[0]
seed_idx = df.index[df['name'] == seed_name][0]

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

print(f"Seed Verbs: {[verbs[idx] for idx in active_s]}")

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
seed_tag_vec = tag_vectors[seed_idx]
seed_tag_norm = tag_norms[seed_idx]
tag_sims = (np.dot(tag_vectors.astype(np.float32), seed_tag_vec.astype(np.float32)) / 
            ((tag_norms + DOT_PRODUCT_LAMBDA) * (seed_tag_norm + DOT_PRODUCT_LAMBDA)))

seed_sem_vec = sem_vectors[seed_idx]
seed_sem_norm = sem_norms[seed_idx]
sem_sims_raw = (np.dot(sem_vectors.astype(np.float32), seed_sem_vec.astype(np.float32)) / 
                (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA))
sem_sims = sem_sims_raw / (seed_sem_norm + SEMANTIC_DOT_PRODUCT_LAMBDA)

seed_topic_dist = topic_distributions[seed_idx]
fz = (seed_topic_dist.astype(np.float32) - topic_means) / (topic_stds + 1e-9)
fz[fz < 0.0] = 0
fn = np.linalg.norm(fz) + 1e-9
fz_unit = fz / fn

topic_sims = np.zeros(N, dtype=np.float32)
batch_size = 100000 
for i in range(0, N, batch_size):
    end = min(i + batch_size, N)
    bz = (topic_distributions[i:end].astype(np.float32) - topic_means) / (topic_stds + 1e-9)
    bz[bz < 0.0] = 0
    bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
    topic_sims[i:end] = np.dot(bz / bn, fz_unit)

# Convert to empirical percentiles instead of Z-scores to ensure a robust [0,1] scale
sem_cdf = pd.Series(sem_sims).rank(pct=True).values.astype(np.float32)
topic_cdf = pd.Series(topic_sims).rank(pct=True).values.astype(np.float32)

print("Blending Vibe and Mech...")
# Now that they are uniform percentiles [0,1], we can softmin them cleanly!
vibe_sim = softmin_blend([sem_cdf, topic_cdf], temperature=0.1)
total_sim = mech_sim * vibe_sim

df['mech_sim'] = mech_sim
df['vibe_sim'] = vibe_sim
df['total_sim'] = total_sim
df['tag_sim'] = tag_sims
df['sem_cdf'] = sem_cdf
df['topic_cdf'] = topic_cdf

results = df[df['appid'] != seed_game['appid']].copy()
results = results.sort_values(by='total_sim', ascending=False).head(20)

print(f"\nTop 20 Similar Games to {seed_name} (Total Time: {time.time() - t0:.2f}s)")
header = f"{'Name':<40} | {'Total':<6} | {'Mech':<6} | {'Vibe':<6} | {'Tag':<6} | {'Sem':<6} | {'Top':<6}"
print(header)
print("-" * len(header))
for i, row in results.iterrows():
    print(f"{row['name'][:40]:<40} | {row['total_sim']:.4f} | {row['mech_sim']:.4f} | {row['vibe_sim']:.4f} | {row['tag_sim']:.4f} | {row['sem_cdf']:.4f} | {row['topic_cdf']:.4f}")

print("\nSpecific Lookup:")
for target in ['A Plague Tale: Requiem', "Hellblade: Senua's Sacrifice", 'The Last of Us™ Part I', 'Brothers - A Tale of Two Sons', 'Dishonored', 'Tomb Raider']:
    match = df[df['name'] == target]
    if not match.empty:
        row = match.iloc[0]
        print(f"{row['name']} -> Total: {row['total_sim']:.4f} | Mech: {row['mech_sim']:.4f} | Vibe: {row['vibe_sim']:.4f} | Tag: {row['tag_sim']:.4f} | Sem: {row['sem_cdf']:.4f} | Topic: {row['topic_cdf']:.4f}")

