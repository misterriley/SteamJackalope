import pandas as pd
import numpy as np
import json
import os

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))

def check_game_pool(seed_name, target_name):
    try:
        idx1 = df[df['name'].str.lower() == seed_name.lower()].index[0]
        idx2 = df[df['name'].str.lower() == target_name.lower()].index[0]
        
        tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
        tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
        sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
        sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
        topic_dist = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_distributions.npy'), mmap_mode='r')
        t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
        t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))

        # Tag Sim
        t_sim = np.dot(tag_vectors[idx1], tag_vectors[idx2]) / ((tag_norms[idx1] + 1.0) * (tag_norms[idx2] + 1.0))
        # Sem Sim
        s_sim = np.dot(sem_vectors[idx1], sem_vectors[idx2]) / ((sem_norms[idx1] + 1.0) * (sem_norms[idx2] + 1.0))
        
        # Topic Sim
        def get_topic_sim(threshold):
            z1 = (topic_dist[idx1] - t_means) / (t_stds + 1e-9)
            z2 = (topic_dist[idx2] - t_means) / (t_stds + 1e-9)
            z1[z1 < threshold] = 0
            z2[z2 < threshold] = 0
            n1 = np.linalg.norm(z1) + 1e-9
            n2 = np.linalg.norm(z2) + 1e-9
            return np.dot(z1 / n1, z2 / n2)

        topic_sim_base = get_topic_sim(2.5)
        topic_sim_rescue = get_topic_sim(1.5)
        
        topic_sim = topic_sim_base
        if (t_sim + s_sim) > 0.1:
            topic_sim = max(topic_sim, topic_sim_rescue)

        def softmin_blend(signals, T=0.01):
            stack = np.stack(signals)
            scaled = -stack / T
            max_val = np.max(scaled)
            exp_vals = np.exp(scaled - max_val)
            weights = exp_vals / np.sum(exp_vals)
            return np.sum(stack * weights)

        consensus = softmin_blend([t_sim, s_sim, topic_sim * 0.1])
        
        return {
            "names": (seed_name, target_name),
            "tag": float(t_sim),
            "sem": float(s_sim),
            "topic": float(topic_sim),
            "consensus": float(consensus)
        }
    except Exception as e:
        return {"error": str(e)}

print(json.dumps(check_game_pool("Spiritfarer®: Farewell Edition", "Neva"), indent=2))
