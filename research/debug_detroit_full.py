import pandas as pd
import numpy as np
import json
import os
import re

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))

def debug_detroit():
    seed_name = "Detroit: Become Human"
    idx1 = df[df['name'] == seed_name].index[0]
    
    tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
    tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
    sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
    sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
    
    NARRATIVE_TAGS = ["Story Rich", "Choices Matter", "Visual Novel", "RPG", "Cinematic", "Multiple Endings", "Interactive Fiction"]
    full_tags_str = df['tags'].fillna('').astype(str).values

    # Vectors
    t_sims = (np.dot(tag_vectors.astype(np.float32), tag_vectors[idx1]) / ((tag_norms[idx1] + 1.0) * (tag_norms + 1.0))) * 11.25
    s_sims = (np.dot(sem_vectors.astype(np.float32), sem_vectors[idx1]) / ((sem_norms[idx1] + 1.0) * (sem_norms + 1.0))) * 10.0
    
    # Simple Topic
    # Skip topic for now, focus on tag/sem/narrative
    consensus = (t_sims + s_sims) / 2.0
    pure = (t_sims * 0.25 + s_sims * 0.25 + consensus * 0.5)

    import ast
    fav_tags_dict = ast.literal_eval(df.iloc[idx1]['tags'])
    active_narr_seed = [t for t in NARRATIVE_TAGS if t in fav_tags_dict]
    pattern = "|".join([rf"'{re.escape(t)}':" for t in active_narr_seed])
    match_series = pd.Series(full_tags_str)
    narr_match_counts = match_series.str.count(pattern).values
    
    pure += np.where(narr_match_counts >= 3, 0.03, 0.0)
    
    # Ranking
    final = pure * 500.0
    final[idx1] = -np.inf
    
    top_indices = np.argsort(-final)[:20]
    print("--- TOP 20 PURE SIMILARITY ---")
    for i in top_indices:
        print(f"{df.iloc[i]['name']} (Pure: {pure[i]:.4f}, Narr: {narr_match_counts[i]})")

debug_detroit()
