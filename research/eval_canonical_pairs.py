import pandas as pd
import numpy as np
import json
import ast
from common.utils import calculate_jackalope_kernel, softmin_blend
from common.constants import (
    TAG_VECTORS_FILE, TAG_NORMS_FILE, 
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, DIFFUSED_VERB_PROFILES_FILE,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)

print("Loading data...")
df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name', 'tags'])
N = len(df)

verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
topic_means = np.load("data/production/topic_means.npy")
topic_stds = np.load("data/production/topic_stds.npy")

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

def eval_game(seed_name, target_name):
    match = df[df['name'] == seed_name]
    if match.empty: return -1, 0.0
    seed_idx = match.index[0]
    seed_appid = match.iloc[0]['appid']
    
    match_target = df[df['name'] == target_name]
    if match_target.empty: return -1, 0.0
    target_appid = match_target.iloc[0]['appid']
    
    total_sim = calculate_jackalope_kernel(
        verb_profiles=verb_profiles,
        seed_verb_profile=verb_profiles[seed_idx],
        sem_vectors=sem_vectors, sem_norms=sem_norms,
        seed_sem_vec=sem_vectors[seed_idx], seed_sem_norm=sem_norms[seed_idx],
        topic_distributions=topic_distributions, seed_topic_dist=topic_distributions[seed_idx],
        topic_means=topic_means, topic_stds=topic_stds,
        tag_scaling_factor=TAG_GLOBAL_SCALING_FACTOR, dot_product_lambda=DOT_PRODUCT_LAMBDA,
        sem_scaling_factor=SEMANTIC_GLOBAL_SCALING_FACTOR, sem_lambda=SEMANTIC_DOT_PRODUCT_LAMBDA
    )
    
    results = pd.DataFrame({'appid': df['appid'], 'score': total_sim})
    results = results[results['appid'] != seed_appid].sort_values(by='score', ascending=False).reset_index(drop=True)
    
    target_rank = results.index[results['appid'] == target_appid].tolist()
    if target_rank:
        return target_rank[0] + 1, results.iloc[target_rank[0]]['score']
    return -1, 0.0

print("\n--- Evaluating 20 Canonical Pairs (PRODUCTION KERNEL) ---")
for g1, g2 in pairs:
    r1, s1 = eval_game(g1, g2)
    r2, s2 = eval_game(g2, g1)
    
    print(f"Pair: {g1[:25]:<25} | {g2[:25]:<25}")
    print(f"  {g1[:15]:<15} -> {g2[:15]:<15} | Rank: {r1:<5} | Score: {s1:.4f}")
    print(f"  {g2[:15]:<15} -> {g1[:15]:<15} | Rank: {r2:<5} | Score: {s2:.4f}")
