import pandas as pd
import numpy as np
import json
import ast
import math

# Load Verbs
with open("tag_categories.json", "r") as f:
    verbs = sorted(json.load(f)["verbs"])

df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'tags'])

verb_to_idx = {v: i for i, v in enumerate(verbs)}
doc_freqs = np.zeros(len(verbs))

total_games = 0

print("Calculating Document Frequencies for verbs...")
for tag_str in df["tags"]:
    if not tag_str or pd.isna(tag_str) or tag_str == "None":
        continue
    try:
        tag_dict = ast.literal_eval(tag_str)
        if isinstance(tag_dict, dict) and tag_dict:
            max_count = max(tag_dict.values())
            # Keep verbs with at least 10% of max count
            active_verbs = [t for t, count in tag_dict.items() if t in verb_to_idx and (count / max_count) >= 0.1]
            if active_verbs:
                total_games += 1
                for v in active_verbs:
                    doc_freqs[verb_to_idx[v]] += 1
    except Exception as e:
        pass

# Calculate IDF: log(N / df)
# We add 1 to avoid log(1) = 0 for the most common tag
idf = np.log(total_games / (doc_freqs + 1)) + 1.0

# Optional: Normalize so max weight is 1.0 (for numerical stability)
# idf = idf / np.max(idf)

idf_dict = {verbs[i]: float(idf[i]) for i in range(len(verbs))}

print(f"\nTotal games with at least one verb: {total_games}")
print("\nTop 5 Most Common (Lowest IDF):")
for v in sorted(idf_dict.items(), key=lambda x: x[1])[:5]:
    print(f"  {v[0]}: IDF {v[1]:.4f} (Freq: {doc_freqs[verb_to_idx[v[0]]]})")

print("\nTop 5 Rarest (Highest IDF):")
for v in sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {v[0]}: IDF {v[1]:.4f} (Freq: {doc_freqs[verb_to_idx[v[0]]]})")

with open("verb_idf_weights.json", "w") as f:
    json.dump(idf_dict, f, indent=2)

print("\nSaved verb_idf_weights.json")
