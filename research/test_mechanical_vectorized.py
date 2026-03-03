import pandas as pd
import numpy as np
import json
import ast
import time

print("Loading data...")
t0 = time.time()

# Load Verbs
with open("tag_categories.json", "r") as f:
    verbs = sorted(json.load(f)["verbs"])
verb_to_idx = {v: i for i, v in enumerate(verbs)}
V = len(verbs)

# Load Diffusion Matrix
S = np.load("data/production/verb_diffusion_matrix.npy")

# Load IDF Weights
with open("verb_idf_weights.json", "r") as f:
    idf_dict = json.load(f)
W = np.array([idf_dict.get(v, 1.0) for v in verbs], dtype=np.float32)

# Load metadata
df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name', 'tags', 'positive', 'negative'])

# Identify Subnautica
seed_game = df[df['name'] == 'Subnautica'].iloc[0]
print(f"Seed Game: {seed_game['name']}")

# Build M matrix
print("Building binary tag matrix M...")
N = len(df)
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

# Populate M (This might take a few seconds)
for i, tag_str in enumerate(df['tags']):
    indices = get_active_verb_indices(tag_str)
    if indices:
        M[i, indices] = 1.0

print(f"Matrix built in {time.time() - t0:.2f} seconds.")

# Subnautica's verbs
active_s = get_active_verb_indices(seed_game['tags'])
print(f"Seed Verbs: {[verbs[idx] for idx in active_s]}")

print("Calculating similarities...")
t1 = time.time()

# Vectorized Chamfer-IDF calculation

# 1. Forward Pass (M -> s)
# sim_to_s[k] is the max similarity from verb k to any verb in s
if active_s:
    sim_to_s = np.max(S[:, active_s], axis=1) # shape: (V,)
else:
    sim_to_s = np.zeros(V)

# Weighted sum of max similarities for each game in M
# M is (N, V), sim_to_s * W is (V,)
# Dot product gives the sum over active verbs in M
score_a_num = M @ (sim_to_s * W)
score_a_den = M @ W

# Avoid division by zero
mask = score_a_den > 0
score_a = np.zeros(N, dtype=np.float32)
score_a[mask] = score_a_num[mask] / score_a_den[mask]

# 2. Backward Pass (s -> M)
# For each verb in s, what is its max similarity to any verb in game i?
sims_b_num = np.zeros(N, dtype=np.float32)
sum_weights_b = 0.0

if active_s:
    for vb in active_s:
        w = W[vb]
        # S[vb, :] is shape (V,). M is (N, V).
        # We want the max similarity of vb to any active verb in M[i, :]
        # By multiplying S[vb, :] by M, we zero out similarities for inactive verbs
        max_sim_b = np.max(M * S[vb, :], axis=1) # shape: (N,)
        sims_b_num += max_sim_b * w
        sum_weights_b += w

score_b = np.zeros(N, dtype=np.float32)
if sum_weights_b > 0:
    score_b = sims_b_num / sum_weights_b

# 3. Total Score
total_score = (score_a + score_b) / 2.0

print(f"Vectorized calculation took {time.time() - t1:.4f} seconds.")

df['mech_sim'] = total_score

# Filter out the seed game and games with no verbs
results = df[(df['appid'] != seed_game['appid']) & (score_a_den > 0)].copy()
results = results.sort_values(by='mech_sim', ascending=False).head(20)

print("\nTop 20 Mechanically Similar Games to Subnautica:")
for i, row in results.iterrows():
    # Only show verbs for context
    v_indices = np.where(M[i] == 1.0)[0]
    v_names = [verbs[idx] for idx in v_indices]
    print(f"{row['mech_sim']:.4f} - {row['name'][:40]:<40} - {v_names}")
