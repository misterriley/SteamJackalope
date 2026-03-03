import pandas as pd
import numpy as np
import json
import ast

with open("tag_categories.json", "r") as f:
    verbs = set(json.load(f)["verbs"])

with open("verb_diffusion_similarities.json", "r") as f:
    diff_sims = json.load(f)

with open("verb_idf_weights.json", "r") as f:
    idf_weights = json.load(f)

def get_verb_sim(v1, v2):
    if v1 == v2: return 1.0
    return diff_sims.get(v1, {}).get(v2, 0.0)

df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name', 'tags'])

def extract_verbs(tag_str):
    if not tag_str or pd.isna(tag_str) or tag_str == "None": return []
    tag_dict = ast.literal_eval(tag_str)
    if not tag_dict: return []
    max_count = max(tag_dict.values())
    return [t for t, count in tag_dict.items() if t in verbs and (count / max_count) >= 0.1]

def calc_mech_sim(verbs_a, verbs_b):
    if not verbs_a or not verbs_b: return 0.0
    
    # Weighted Chamfer
    sims_a_to_b = []
    weights_a = []
    for va in verbs_a:
        max_sim = max([get_verb_sim(va, vb) for vb in verbs_b])
        w = idf_weights.get(va, 1.0)
        sims_a_to_b.append(max_sim * w)
        weights_a.append(w)
        
    sims_b_to_a = []
    weights_b = []
    for vb in verbs_b:
        max_sim = max([get_verb_sim(vb, va) for va in verbs_a])
        w = idf_weights.get(vb, 1.0)
        sims_b_to_a.append(max_sim * w)
        weights_b.append(w)
        
    score_a = np.sum(sims_a_to_b) / np.sum(weights_a) if weights_a else 0.0
    score_b = np.sum(sims_b_to_a) / np.sum(weights_b) if weights_b else 0.0
        
    return (score_a + score_b) / 2.0

pairs = [
    ("Hollow Knight", "Ori and the Will of the Wisps"),
    ("Portal 2", "The Talos Principle"),
    ("DOOM", "Counter-Strike 2"),
    ("Slay the Spire", "Monster Train"),
    ("Divinity: Original Sin 2 - Definitive Edition", "Baldur's Gate 3")
]

target_names = [name for pair in pairs for name in pair]
games_df = df[df['name'].isin(target_names)]

game_verbs = {}
for _, row in games_df.iterrows():
    game_verbs[row['name']] = extract_verbs(row['tags'])

names = [n for n in target_names if n in game_verbs]

print("\n--- Cross-Similarity Matrix (IDF Weighted) ---")
header = f"{'Game':<45} | " + " | ".join([f"{name[:10]:<10}" for name in names])
print(header)
print("-" * len(header))

for name1 in names:
    row_str = f"{name1:<45} | "
    for name2 in names:
        sim = calc_mech_sim(game_verbs[name1], game_verbs[name2])
        row_str += f"{sim:10.3f} | "
    print(row_str)
