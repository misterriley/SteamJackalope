import pandas as pd
import numpy as np
import json
import os
import re

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))

def softmin_blend(signals, T=0.01):
    stack = np.stack(signals)
    scaled = -stack / T
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals)
    return np.sum(stack * weights)

def check_detroit_v3():
    seed_name = "Detroit: Become Human"
    targets = [
        "Retropolis 2: Never Say Goodbye",
        "AN IOU",
        "Kunado Chronicles",
        "The Uncertain: Light At The End",
        "Beyond: Two Souls",
        "Dreamfall: The Longest Journey"
    ]
    
    idx1 = df[df['name'] == seed_name].index[0]
    
    tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
    tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
    sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
    sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
    topic_dist = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_distributions.npy'), mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))
    
    NARRATIVE_TAGS = ["Story Rich", "Choices Matter", "Visual Novel", "RPG", "Cinematic", "Multiple Endings", "Interactive Fiction"]

    res = []
    for target_name in targets:
        try:
            idx2 = df[df['name'] == target_name].index[0]
            
            t_sim = np.dot(tag_vectors[idx1], tag_vectors[idx2]) / ((tag_norms[idx1] + 1.0) * (tag_norms[idx2] + 1.0))
            s_sim = np.dot(sem_vectors[idx1], sem_vectors[idx2]) / ((sem_norms[idx1] + 1.0) * (sem_norms[idx2] + 1.0))
            
            def get_topic_sim(threshold):
                z1 = (topic_dist[idx1] - t_means) / (t_stds + 1e-9)
                z2 = (topic_dist[idx2] - t_means) / (t_stds + 1e-9)
                z1[z1 < threshold] = 0
                z2[z2 < threshold] = 0
                n1 = np.linalg.norm(z1) + 1e-9
                n2 = np.linalg.norm(z2) + 1e-9
                return np.dot(z1 / n1, z2 / n2)

            top_sim = get_topic_sim(2.5)
            if (t_sim + s_sim) > 0.05:
                top_sim = max(top_sim, get_topic_sim(1.5))

            consensus = softmin_blend([t_sim, s_sim, top_sim * 0.1])
            pure = (t_sim * 0.25 + s_sim * 0.25 + consensus * 0.5)
            
            # Rescue
            import ast
            fav_tags_dict = ast.literal_eval(df.iloc[idx1]['tags'])
            target_tags_dict = ast.literal_eval(df.iloc[idx2]['tags'])
            
            active_narr_seed = [t for t in NARRATIVE_TAGS if t in fav_tags_dict]
            narr_match_count = sum([1 for t in active_narr_seed if t in target_tags_dict])
            
            if len(active_narr_seed) >= 2:
                if narr_match_count >= 3: pure += 0.03
                if narr_match_count >= 4: consensus = max(consensus, 0.01)

            res.append({
                "target": target_name,
                "pure": float(pure),
                "narr_matches": narr_match_count,
                "consensus": float(consensus)
            })
        except Exception as e:
            res.append({"target": target_name, "error": str(e)})
            
    return res

print(json.dumps(check_detroit_v3(), indent=2))
