import pandas as pd
import numpy as np
import json
import ast
from scipy.stats import norm
from common.utils import softmin_blend
from common.constants import (
    TAG_VECTORS_FILE, TAG_NORMS_FILE, 
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE,
    SEMANTIC_SIMILARITY_MEAN, SEMANTIC_SIMILARITY_STD,
    TOPIC_SIMILARITY_MEAN, TOPIC_SIMILARITY_STD,
    SEMANTIC_DOT_PRODUCT_LAMBDA
)

print("Loading data...")
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

# Weighted Diffused Profiles
MW = M * W
Profiles = MW @ S

sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
topic_means = np.load("data/production/topic_means.npy")
topic_stds = np.load("data/production/topic_stds.npy")

bz = (topic_distributions.astype(np.float32) - topic_means) / (topic_stds + 1e-9)
bz[bz < 0.0] = 0
bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
bz_unit = bz / bn

pairs = [
    ("A Plague Tale: Innocence", "A Plague Tale: Requiem"),
    ("Portal", "Portal 2"),
    ("Detroit: Become Human", "Beyond: Two Souls"),
    ("Half-Life", "Half-Life 2"),
    ("Left 4 Dead", "Left 4 Dead 2"),
    ("Divinity: Original Sin - Enhanced Edition", "Divinity: Original Sin 2 - Definitive Edition"),
    ("Subnautica", "Subnautica: Below Zero"),
    ("DOOM", "DOOM Eternal"),
    ("Ori and the Blind Forest: Definitive Edition", "Ori and the Will of the Wisps"),
    ("Spelunky", "Spelunky 2"),
    ("Risk of Rain (2013)", "Risk of Rain 2"),
    ("Dishonored", "Dishonored 2"),
    ("BioShock™ Remastered", "BioShock™ 2 Remastered"),
    ("Borderlands 2", "Borderlands 3"),
    ("The Witcher 2: Assassins of Kings Enhanced Edition", "The Witcher 3: Wild Hunt"),
    ("XCOM: Enemy Unknown", "XCOM® 2"),
    ("Nioh: Complete Edition", "Nioh 2 - The Complete Edition"),
    ("Fallout: A Post Nuclear Role Playing Game", "Fallout 2: A Post Nuclear Role Playing Game"),
    ("Darksiders Warmastered Edition", "Darksiders II Deathinitive Edition"),
    ("DARK SOULS™: REMASTERED", "DARK SOULS™ II: Scholar of the First Sin")
]

ALPHA = 0.1

def eval_game(seed_name, target_name):
    match = df[df['name'] == seed_name]
    if match.empty: return -1, 0.0, 0.0, 0.0
    seed_idx = match.index[0]
    seed_appid = match.iloc[0]['appid']
    
    match_target = df[df['name'] == target_name]
    if match_target.empty: return -1, 0.0, 0.0, 0.0
    target_appid = match_target.iloc[0]['appid']
    
    # 1. Mech Sim (Softened Fuzzy Jaccard)
    seed_profile = Profiles[seed_idx]
    
    intersection = np.sum(np.minimum(Profiles, seed_profile), axis=1)
    union = np.sum(np.maximum(Profiles, seed_profile), axis=1)
    
    # Soft Jaccard = I / (I + alpha * (U - I))
    denominator = intersection + ALPHA * (union - intersection)
    
    mech_sim = np.zeros(N, dtype=np.float32)
    mask = denominator > 0
    mech_sim[mask] = intersection[mask] / denominator[mask]
    
    # 2. Vibe Sim
    seed_sem_vec = sem_vectors[seed_idx]
    seed_sem_norm = sem_norms[seed_idx]
    sem_sims_raw = (np.dot(sem_vectors.astype(np.float32), seed_sem_vec.astype(np.float32)) / 
                    (sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA))
    sem_sims = sem_sims_raw / (seed_sem_norm + SEMANTIC_DOT_PRODUCT_LAMBDA)
    
    fz_unit_seed = bz_unit[seed_idx]
    topic_sims = np.dot(bz_unit, fz_unit_seed)
    
    sem_cdf = pd.Series(sem_sims).rank(pct=True).values.astype(np.float32)
    topic_cdf = pd.Series(topic_sims).rank(pct=True).values.astype(np.float32)
    
    vibe_sim = softmin_blend([sem_cdf, topic_cdf], temperature=0.1)
    
    # Combine Multiplicative
    total_sim = mech_sim * vibe_sim
    
    results = pd.DataFrame({'appid': df['appid'], 'score': total_sim})
    results = results[results['appid'] != seed_appid].sort_values(by='score', ascending=False).reset_index(drop=True)
    
    target_rank = results.index[results['appid'] == target_appid].tolist()
    if target_rank:
        t_idx = df.index[df['appid'] == target_appid][0]
        return target_rank[0] + 1, results.iloc[target_rank[0]]['score'], mech_sim[t_idx], vibe_sim[t_idx]
    return -1, 0.0, 0.0, 0.0

print("\n--- Evaluating 20 Canonical Pairs (Soft Fuzzy Jaccard + Multiplicative) ---")
for g1, g2 in pairs:
    r1, s1, m1, v1 = eval_game(g1, g2)
    r2, s2, m2, v2 = eval_game(g2, g1)
    
    print(f"Pair: {g1[:25]:<25} | {g2[:25]:<25}")
    print(f"  {g1[:15]:<15} -> {g2[:15]:<15} | Rank: {r1:<5} | Score: {s1:.4f} (Mech: {m1:.4f}, Vibe: {v1:.4f})")
    print(f"  {g2[:15]:<15} -> {g1[:15]:<15} | Rank: {r2:<5} | Score: {s2:.4f} (Mech: {m2:.4f}, Vibe: {v2:.4f})")
