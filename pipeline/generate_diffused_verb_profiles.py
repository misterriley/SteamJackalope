import pandas as pd
import numpy as np
import json
import ast

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

np.save("data/production/diffused_verb_profiles.npy", Profiles.astype(np.float16))
print("Saved data/production/diffused_verb_profiles.npy")
