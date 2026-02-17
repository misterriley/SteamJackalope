import numpy as np
import json

W = np.load('data/production/w_tag.npy').astype(np.float32)
unique_tags = json.load(open('data/production/tag_names.json'))
tag_to_idx = {t: i for i, t in enumerate(unique_tags)}

def tag_vec(name):
    idx = tag_to_idx.get(name)
    if idx is None:
        print(f"Tag {name} not found!")
        return None
    return W[idx]

def sim(n1, n2):
    v1 = tag_vec(n1)
    v2 = tag_vec(n2)
    if v1 is None or v2 is None: return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Sci-fi vs Mahjong:", sim('Sci-fi', 'Mahjong'))
print("Sci-fi vs NSFW:", sim('Sci-fi', 'NSFW'))
print("Sci-fi vs Space:", sim('Sci-fi', 'Space'))
print("Mahjong vs NSFW:", sim('Mahjong', 'NSFW'))
