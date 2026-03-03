import pandas as pd
import numpy as np
import json
import os

PRODUCTION_DATA_DIR = 'data/production'

df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))
tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
topic_dist = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_distributions.npy'), mmap_mode='r')
t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))

def get_sims(seed_name, target_name):
    try:
        idx1 = df[df['name'].str.lower() == seed_name.lower()].index[0]
        idx2 = df[df['name'].str.lower() == target_name.lower()].index[0]
        
        # Tag Similarity
        t_sim = np.dot(tag_vectors[idx1], tag_vectors[idx2]) / ((tag_norms[idx1] + 1.0) * (tag_norms[idx2] + 1.0))
        
        # Semantic Similarity
        s_sim = np.dot(sem_vectors[idx1], sem_vectors[idx2]) / ((sem_norms[idx1] + 1.0) * (sem_norms[idx2] + 1.0))
        
        # Topic Similarity (Consensus logic - 2.5Z)
        z1 = (topic_dist[idx1] - t_means) / (t_stds + 1e-9)
        z2 = (topic_dist[idx2] - t_means) / (t_stds + 1e-9)
        z1[z1 < 2.5] = 0
        z2[z2 < 2.5] = 0
        norm1 = np.linalg.norm(z1) + 1e-9
        norm2 = np.linalg.norm(z2) + 1e-9
        topic_sim = np.dot(z1 / norm1, z2 / norm2)
        
        # Blended Logic
        consensus = (t_sim + s_sim + topic_sim*0.1) / 3.0 # Simplified
        pure_thematic = (t_sim + consensus) / 2.0
        
        return {
            "seed": seed_name,
            "target": target_name,
            "tag_sim": float(t_sim),
            "sem_sim": float(s_sim),
            "topic_sim": float(topic_sim),
            "consensus": float(consensus),
            "pure_thematic": float(pure_thematic)
        }
    except Exception as e:
        return {"error": str(e), "names": (seed_name, target_name)}

checks = [
    ("Antichamber", "Super reaKtor"),
    ("Antichamber", "Super Dungeon Boy"),
    ("Antichamber", "Protect Your Fool"),
    ("Ori and the Will of the Wisps", "Deltarune")
]

print(json.dumps([get_sims(s, t) for s, t in checks], indent=2))
