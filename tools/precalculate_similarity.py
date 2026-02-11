
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_TAG_FILE,
    TAG_VECTORS_FILE,
    METADATA_FILE,
    DOT_PRODUCT_LAMBDA,
    EPSILON,
    SIMILARITY_LISTS_FILE
)

def precalculate_similarity_lists():
    print("Loading data for precalculation...")
    metadata = pd.read_parquet(METADATA_FILE)
    tag_vectors = np.load(TAG_VECTORS_FILE)
    embeddings_structural = np.load(EMBEDDINGS_TAG_FILE)
    
    # Normalize semantic structural vectors
    def normalize(m):
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = EPSILON
        return m / norms
    
    embeddings_structural_norm = normalize(embeddings_structural)
    tag_vectors_norms = np.linalg.norm(tag_vectors, axis=1)

    # 1. Find 20 most popular games with low similarity
    print("Finding seed games...")
    metadata['total_reviews'] = metadata['positive'] + metadata['negative']
    sorted_by_pop = metadata.sort_values('total_reviews', ascending=False)
    
    seeds = []
    seed_indices = []
    
    for idx, row in sorted_by_pop.iterrows():
        if len(seeds) >= 20:
            break
            
        current_tag_vec = tag_vectors[idx]
        current_norm = tag_vectors_norms[idx]
        
        is_diverse = True
        for s_idx in seed_indices:
            # Penalized cosine similarity
            dot = np.dot(current_tag_vec, tag_vectors[s_idx])
            denom = (current_norm * tag_vectors_norms[s_idx]) + DOT_PRODUCT_LAMBDA
            sim = dot / (denom if denom > 0 else EPSILON)
            
            if sim >= 0.1:
                is_diverse = False
                break
        
        if is_diverse:
            seeds.append(row['name'])
            seed_indices.append(idx)
            print(f"Found seed {len(seeds)}: {row['name']} (Reviews: {row['total_reviews']})")

    # 2. Find 5 most similar for each seed (Tags and Semantic)
    print("Finding similar games...")
    results_tags = []
    results_semantic = []
    
    for s_idx in seed_indices:
        seed_name = metadata.iloc[s_idx]['name']
        seed_appid = metadata.iloc[s_idx]['appid']
        
        # Tags similarity
        tag_dot = np.dot(tag_vectors, tag_vectors[s_idx])
        tag_denom = (tag_vectors_norms * tag_vectors_norms[s_idx]) + DOT_PRODUCT_LAMBDA
        tag_sims = tag_dot / np.where(tag_denom > 0, tag_denom, EPSILON)
        tag_sims[s_idx] = -1 # Exclude self
        
        top_tag_indices = np.argsort(-tag_sims)[:5]
        top_tag_games = []
        for t_idx in top_tag_indices:
            top_tag_games.append({
                'name': metadata.iloc[t_idx]['name'],
                'appid': int(metadata.iloc[t_idx]['appid']),
                'similarity': float(tag_sims[t_idx])
            })
        
        results_tags.append({
            'seed_name': seed_name,
            'seed_appid': int(seed_appid),
            'similar': top_tag_games
        })
        
        # Semantic similarity
        sem_sims = np.dot(embeddings_structural_norm, embeddings_structural_norm[s_idx])
        sem_sims[s_idx] = -1 # Exclude self
        
        top_sem_indices = np.argsort(-sem_sims)[:5]
        top_sem_games = []
        for t_idx in top_sem_indices:
            top_sem_games.append({
                'name': metadata.iloc[t_idx]['name'],
                'appid': int(metadata.iloc[t_idx]['appid']),
                'similarity': float(sem_sims[t_idx])
            })
            
        results_semantic.append({
            'seed_name': seed_name,
            'seed_appid': int(seed_appid),
            'similar': top_sem_games
        })

    # Save results
    import json
    output = {
        'tags': results_tags,
        'semantic': results_semantic
    }
    
    with open(SIMILARITY_LISTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {SIMILARITY_LISTS_FILE}")

if __name__ == "__main__":
    precalculate_similarity_lists()
