import pandas as pd
import numpy as np
import json
import ast

# Load Verbs
with open("tag_categories.json", "r") as f:
    verbs = set(json.load(f)["verbs"])

# Load Diffusion Matrix
with open("verb_diffusion_similarities.json", "r") as f:
    diff_sims = json.load(f)

# The matrix is indexed by sorted verbs, let's just use the dict for fast lookup
def get_verb_sim(v1, v2):
    if v1 == v2: return 1.0
    return diff_sims.get(v1, {}).get(v2, 0.0)

# Load metadata
df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name', 'tags'])

witcher = df[df['appid'] == 292030].iloc[0]
skyrim = df[df['appid'] == 489830].iloc[0]

def extract_verbs(tag_str):
    if not tag_str or pd.isna(tag_str) or tag_str == "None": return []
    tag_dict = ast.literal_eval(tag_str)
    if not tag_dict: return []
    max_count = max(tag_dict.values())
    # Keep verbs with at least 10% of max count
    return [t for t, count in tag_dict.items() if t in verbs and (count / max_count) >= 0.1]

w_verbs = extract_verbs(witcher['tags'])
s_verbs = extract_verbs(skyrim['tags'])

print(f"Witcher Verbs: {w_verbs}")
print(f"Skyrim Verbs: {s_verbs}")

# Calculate Chamfer-like Mechanical Similarity
# For every verb in Game A, find the max similarity to any verb in Game B.
# Then do the same for Game B to Game A. Average the two sets.

def calc_mech_sim(verbs_a, verbs_b):
    if not verbs_a or not verbs_b: return 0.0
    
    sims_a_to_b = []
    for va in verbs_a:
        max_sim = max([get_verb_sim(va, vb) for vb in verbs_b])
        sims_a_to_b.append(max_sim)
        
    sims_b_to_a = []
    for vb in verbs_b:
        max_sim = max([get_verb_sim(vb, va) for va in verbs_a])
        sims_b_to_a.append(max_sim)
        
    print("\nWitcher -> Skyrim (Max Verb Matches):")
    for va, max_s in zip(verbs_a, sims_a_to_b):
        best_match = max(verbs_b, key=lambda vb: get_verb_sim(va, vb))
        print(f"  {va} -> {best_match} ({max_s:.4f})")
        
    print("\nSkyrim -> Witcher (Max Verb Matches):")
    for vb, max_s in zip(verbs_b, sims_b_to_a):
        best_match = max(verbs_a, key=lambda va: get_verb_sim(vb, va))
        print(f"  {vb} -> {best_match} ({max_s:.4f})")
        
    return (np.mean(sims_a_to_b) + np.mean(sims_b_to_a)) / 2.0

total_sim = calc_mech_sim(w_verbs, s_verbs)
print(f"\nTotal Mechanical Similarity: {total_sim:.4f}")
