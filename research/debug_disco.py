import pandas as pd
import numpy as np
import json
import os
import re
import ast

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))

def softmin_blend(signals, T=0.01):
    stack = np.stack(signals)
    scaled = -stack / T
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals)
    return np.sum(stack * weights)

def debug_disco():
    seed_appid = 632470 # Disco Elysium
    target_appids = [2760650, 917720, 370910]
    
    idx1 = df[df['appid'] == seed_appid].index[0]
    
    tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
    tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
    sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
    sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
    topic_dist = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_distributions.npy'), mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_means.npy'))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, 'topic_stds.npy'))
    
    NARRATIVE_TAGS = ["Story Rich", "Choices Matter", "Visual Novel", "RPG", "Cinematic", "Multiple Endings", "Interactive Fiction", "Female Protagonist", "Philosophical"]
    ACTION_TAGS = ["Action", "Combat", "Action-Adventure", "Shooter", "FPS"]
    RPG_ACTION_TAGS = ["Action RPG", "Third-Person Shooter", "TPS", "Character Action Game"]
    RPG_CRPG_TAGS = ["CRPG", "Isometric", "Turn-Based Combat", "Real Time Tactics", "Party-Based RPG"]

    fav_tags_dict = ast.literal_eval(df.iloc[idx1]['tags'])
    max_tag_val = max(fav_tags_dict.values())
    active_narr_seed = [t for t in NARRATIVE_TAGS if t in fav_tags_dict]

    res = []
    for aid in target_appids:
        try:
            idx2 = df[df['appid'] == aid].index[0]
            target_name = df.iloc[idx2]['name']
            
            t_sim = (np.dot(tag_vectors[idx1], tag_vectors[idx2]) / ((tag_norms[idx1] + 1.0) * (tag_norms[idx2] + 1.0))) * 11.25
            s_sim = (np.dot(sem_vectors[idx1], sem_vectors[idx2]) / ((sem_norms[idx1] + 1.0) * (sem_norms[idx2] + 1.0))) * 10.0
            
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
            target_tags_dict = ast.literal_eval(df.iloc[idx2]['tags'])
            narr_match_count = sum([1 for t in active_narr_seed if t in target_tags_dict])
            
            if len(active_narr_seed) >= 2:
                if narr_match_count >= 3: pure += 0.03
                if narr_match_count >= 4: consensus = max(consensus, 0.01)

            # Cinematic Resonance
            is_cinematic_seed = "Cinematic" in fav_tags_dict and "Story Rich" in fav_tags_dict
            if is_cinematic_seed:
                if "Cinematic" in target_tags_dict or "Story Rich" in target_tags_dict:
                    pure += 0.05

            # Vetoes logic (mocked here for specific check)
            vetoed = pure
            # RPG style
            is_action_rpg = any(t in fav_tags_dict for t in RPG_ACTION_TAGS)
            is_crpg = any(t in fav_tags_dict for t in RPG_CRPG_TAGS)
            target_is_crpg = any(t in target_tags_dict for t in RPG_CRPG_TAGS)
            target_is_action_rpg = any(t in target_tags_dict for t in RPG_ACTION_TAGS)
            
            if is_crpg and target_is_action_rpg: vetoed *= 0.05
            if is_action_rpg and target_is_crpg: vetoed *= 0.05
            
            # Consensus Veto
            pass_cons = consensus >= (0.001 if len(active_narr_seed) >= 2 else 0.005)

            res.append({
                "target": target_name,
                "pure": float(pure),
                "vetoed": float(vetoed),
                "consensus": float(consensus),
                "pass_cons": bool(pass_cons),
                "narr_matches": narr_match_count,
                "is_crpg": target_is_crpg
            })
        except Exception as e:
            res.append({"appid": aid, "error": str(e)})
            
    return res

print(json.dumps(debug_disco(), indent=2))
