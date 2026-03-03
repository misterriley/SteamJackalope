import json
import ast
import pandas as pd
import numpy as np
import os

# Load verb categories
with open("tag_categories.json", "r") as f:
    categories = json.load(f)
verbs = sorted(categories["verbs"])

print(f"Loaded {len(verbs)} verbs.")

# Load metadata
print("Loading metadata...")
df = pd.read_parquet("data/production/metadata.parquet", columns=["appid", "tags"])

# Build binary matrix for verbs
# Rows = games, Cols = verbs
verb_to_idx = {v: i for i, v in enumerate(verbs)}
matrix = np.zeros((len(df), len(verbs)), dtype=np.float32)

print("Building tag matrix...")
for i, tag_str in enumerate(df["tags"]):
    if not tag_str or pd.isna(tag_str) or tag_str == "None":
        continue
    try:
        tag_dict = ast.literal_eval(tag_str)
        if isinstance(tag_dict, dict) and tag_dict:
            max_count = max(tag_dict.values())
            for t, count in tag_dict.items():
                # Apply a small threshold (10% of max tag count) to filter out noise
                if t in verb_to_idx and (count / max_count) >= 0.1:
                    matrix[i, verb_to_idx[t]] = 1.0  # binary presence
    except Exception as e:
        pass

print("Calculating Jaccard Similarity...")
# Calculate Jaccard Similarity
# Jaccard(A, B) = |A & B| / |A U B|
intersection = matrix.T @ matrix
col_sums = matrix.sum(axis=0)
union = col_sums[:, None] + col_sums[None, :] - intersection

jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)

print("Calculating Pearson Correlation...")
# Calculate Pearson Correlation
# np.corrcoef expects variables as rows, so we transpose
correlation = np.corrcoef(matrix, rowvar=False)

# Create a dictionary for saving top similarities
results = {}
for i, v1 in enumerate(verbs):
    results[v1] = {}
    for j, v2 in enumerate(verbs):
        if i != j:
            results[v1][v2] = {
                "jaccard": round(float(jaccard[i, j]), 4),
                "correlation": round(float(correlation[i, j]), 4)
            }

with open("verb_similarities.json", "w") as f:
    json.dump(results, f, indent=2)

np.save("data/production/verb_jaccard_matrix.npy", jaccard)
np.save("data/production/verb_correlation_matrix.npy", correlation)

print("Saved verb_similarities.json and npy matrices.")
