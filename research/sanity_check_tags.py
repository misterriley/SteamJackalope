import numpy as np
import pandas as pd

import ast

def sanity_check():
    print("Loading data...")
    metadata = pd.read_parquet("metadata.parquet")
    tag_vectors = np.load("steam_tag_vectors.npy")
    
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)

    # 1. Identify 20 games with a lot of tags
    print("Identifying 20 games with many tags...")
    
    # Helper to count tags in a stringified dict
    def count_tags(tag_str):
        if not tag_str or tag_str == '[]': return 0
        try:
            d = ast.literal_eval(tag_str)
            return len(d) if isinstance(d, dict) else 0
        except:
            return 0

    metadata['tag_count'] = metadata['tags'].apply(count_tags)
    
    # 1. Identify 1000 games with a lot of tags
    print("Identifying 1000 games with many tags...")
    # Drop duplicates by name to avoid 1.0 similarity for same game versions
    metadata_clean = metadata.drop_duplicates(subset=['name'])
    top_1000 = metadata_clean.sort_values('tag_count', ascending=False).head(1000)
    indices = top_1000.index.tolist()
    names = top_1000['name'].tolist()
    vectors = tag_vectors[indices]
    
    # 2. Generate Cosine Similarity Matrix
    print("Calculating similarities for 1,000,000 pairs...")
    # Normalize vectors for fast cosine similarity via dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    v_norm = vectors / norms
    
    sim_matrix = np.dot(v_norm, v_norm.T)
    
    print("\nFinding top and bottom similarities (excluding self-similarity)...")
    pairs = []
    # Only take upper triangle to avoid duplicates and self-similarity
    rows, cols = np.triu_indices(1000, k=1)
    
    for r, c in zip(rows, cols):
        pairs.append((names[r], names[c], sim_matrix[r, c]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTOP 20 MOST SIMILAR PAIRS:")
    for p in pairs[:20]:
        print(f"{p[0]} vs {p[1]}: {p[2]:.4f}")

    print("\nBOTTOM 20 LEAST SIMILAR PAIRS:")
    for p in pairs[-20:]:
        print(f"{p[0]} vs {p[1]}: {p[2]:.4f}")

if __name__ == "__main__":
    sanity_check()
