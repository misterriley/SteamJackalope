import pandas as pd
import numpy as np
import ast
import json
import os
import re
from common.constants import (
    METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)
from common.utils import calculate_jackalope_kernel

print("Loading data...")
df = pd.read_parquet(METADATA_FILE)
verb_profiles = np.load(DIFFUSED_VERB_PROFILES_FILE, mmap_mode='r')
sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
topic_distributions = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
topic_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
topic_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)

def get_info(appid):
    m = df[df['appid'] == appid]
    if m.empty: return None
    return m.index[0], m.iloc[0]['name'], m.iloc[0]['tags']

frog_idx, frog_name, frog_tags = get_info(1194840)
algebra_idx, algebra_name, algebra_tags = get_info(1379510)

print(f"\nSeed: {frog_name} ({1194840})")
print(f"Target: {algebra_name} ({1379510})")

def get_sim_breakdown(s_idx, t_idx):
    sim, comps = calculate_jackalope_kernel(
        verb_profiles=verb_profiles,
        seed_verb_profile=verb_profiles[s_idx],
        sem_vectors=sem_vectors, sem_norms=sem_norms,
        seed_sem_vec=sem_vectors[s_idx], seed_sem_norm=sem_norms[s_idx],
        topic_distributions=topic_distributions, seed_topic_dist=topic_distributions[s_idx],
        topic_means=topic_means, topic_stds=topic_stds,
        tag_scaling_factor=1.0, dot_product_lambda=1.0,
        sem_scaling_factor=1.0, sem_lambda=0.01,
        return_components=True
    )
    # We need to extract the target's value from the full arrays
    return sim[t_idx], {k: v[t_idx] if isinstance(v, np.ndarray) else v for k, v in comps.items()}

print("\n--- Algebra Ridge Breakdown ---")
score, comps = get_sim_breakdown(frog_idx, algebra_idx)
print(f"Score: {score:.4f}")
print(f"Components: {comps}")

# Find a "Frog" match that is ranking high
# We'll calculate all and find the top one with 'Frog' in name
total_sim = calculate_jackalope_kernel(
    verb_profiles=verb_profiles,
    seed_verb_profile=verb_profiles[frog_idx],
    sem_vectors=sem_vectors, sem_norms=sem_norms,
    seed_sem_vec=sem_vectors[frog_idx], seed_sem_norm=sem_norms[frog_idx],
    topic_distributions=topic_distributions, seed_topic_dist=topic_distributions[frog_idx],
    topic_means=topic_means, topic_stds=topic_stds,
    tag_scaling_factor=1.0, dot_product_lambda=1.0,
    sem_scaling_factor=1.0, sem_lambda=0.01
)

df_scores = pd.DataFrame({'appid': df['appid'], 'name': df['name'], 'score': total_sim})
frog_matches = df_scores[df_scores['name'].str.contains('Frog', case=False) & (df_scores['appid'] != 1194840)]
top_frog = frog_matches.sort_values(by='score', ascending=False).iloc[0]

print(f"\n--- Top 'Frog' Match: {top_frog['name']} ({top_frog['appid']}) ---")
f_idx = df[df['appid'] == top_frog['appid']].index[0]
score_f, comps_f = get_sim_breakdown(frog_idx, f_idx)
print(f"Score: {score_f:.4f}")
print(f"Components: {comps_f}")

def get_raw_jaccard(s_tags, t_tags):
    if isinstance(s_tags, str): s_tags = ast.literal_eval(s_tags)
    if isinstance(t_tags, str): t_tags = ast.literal_eval(t_tags)
    s_set = set(s_tags.keys())
    t_set = set(t_tags.keys())
    if not s_set or not t_set: return 0.0
    return len(s_set & t_set) / len(s_set | t_set)

print(f"Algebra Ridge Raw Jaccard: {get_raw_jaccard(frog_tags, algebra_tags):.4f}")

# Find Amazing Frog tags
amazing_tags = df[df['appid'] == top_frog['appid']].iloc[0]['tags']
def detect_hijack(name_s, name_t, tags_s, tags_t, sem_sim):
    jaccard = get_raw_jaccard(tags_s, tags_t)
    
    # Simple keyword extraction (words > 3 chars)
    def get_keywords(name):
        return set(re.findall(r'\b\w{4,}\b', name.lower()))
    
    kw_s = get_keywords(name_s)
    kw_t = get_keywords(name_t)
    shared = kw_s & kw_t
    
    # Common words to ignore
    STOPWORDS = {'edition', 'game', 'decade', 'remaster', 'deluxe', 'pack', 'collection'}
    shared = shared - STOPWORDS
    
    if shared and jaccard < 0.2 and sem_sim > 0.3:
        return True, shared, jaccard
    return False, shared, jaccard

print("\n--- Hijack Detection ---")
h_a, s_a, j_a = detect_hijack(frog_name, algebra_name, frog_tags, algebra_tags, 0.21)
print(f"Algebra Ridge Hijack: {h_a} (Shared: {s_a}, Jaccard: {j_a:.4f})")

score_f_raw_sem = 0.358 # from previous run
h_f, s_f, j_f = detect_hijack(frog_name, top_frog['name'], frog_tags, amazing_tags, score_f_raw_sem)
print(f"Amazing Frog? Hijack: {h_f} (Shared: {s_f}, Jaccard: {j_f:.4f})")
with open("tag_categories.json", "r") as f:
    verbs = sorted(json.load(f)["verbs"])

def print_verbs(idx, name):
    prof = verb_profiles[idx]
    active = [(verbs[i], prof[i]) for i in range(len(verbs)) if prof[i] > 0.5]
    active = sorted(active, key=lambda x: x[1], reverse=True)[:10]
    print(f"{name} Top Verbs: {active}")

print_verbs(frog_idx, frog_name)
print_verbs(algebra_idx, algebra_name)
print_verbs(f_idx, top_frog['name'])
