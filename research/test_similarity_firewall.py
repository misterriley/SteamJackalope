import pandas as pd
import numpy as np
import json

def normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms

def get_list(val):
    if pd.isna(val).all() if hasattr(val, 'all') else pd.isna(val): return []
    if isinstance(val, dict): return list(val.keys())
    if isinstance(val, str):
        try:
            d = eval(val)
            if isinstance(d, dict): return list(d.keys())
        except: pass
        return [x.strip() for x in val.split(',')]
    if hasattr(val, 'tolist'): return val.tolist()
    if isinstance(val, np.ndarray) and len(val.shape) == 0 and isinstance(val.item(), dict):
        return list(val.item().keys())
    return list(val)

def identify_puzzle_subgenre(tags):
    tags_set = set(tags)
    if 'Hidden Object' in tags_set: return 'Hidden Object'
    if 'Automation' in tags_set or 'Programming' in tags_set: return 'Automation'
    if 'Sokoban' in tags_set or 'Grid-Based Movement' in tags_set: return 'Sokoban/Grid'
    if 'Puzzle' in tags_set and ('First-Person' in tags_set or '3D Platformer' in tags_set or 'Open World' in tags_set): return 'Spatial/3D'
    return 'Generic/Other'

def evaluate_similarity(target_name):
    print(f"Loading data to evaluate {target_name}...")
    df = pd.read_parquet('data/production/metadata.parquet')
    features = {}
    features['tags'] = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    features['desc'] = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    features['verbs'] = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    features['graph'] = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))

    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}

    matches = df[df['name'] == target_name]
    if len(matches) == 0: matches = df[df['name'].str.contains(target_name, case=False, na=False)]
    if len(matches) == 0:
        print(f"Game '{target_name}' not found.")
        return

    target_idx = matches.index[0]
    t_row = df.loc[target_idx]
    t_subgenre = identify_puzzle_subgenre(get_list(t_row.get('tags', [])))

    sims = {}
    for k in ['tags', 'desc', 'verbs', 'graph']:
        t_vec = features[k][target_idx].reshape(1, -1)
        sims[k] = np.dot(features[k], t_vec.T).flatten()
        
    graph_sim_discounted = sims['graph'] * np.where(df['pop_z'] > 0, np.exp(-0.15 * df['pop_z']), 1.0)

    total_sim = (
        weights['tags'] * sims['tags'] +
        weights['desc'] * sims['desc'] +
        weights['verbs'] * sims['verbs'] +
        weights['graph'] * graph_sim_discounted
    )

    total_reviews = df['positive'].fillna(0) + df['negative'].fillna(0)
    pos_ratio = df['positive'].fillna(0) / (total_reviews + 1e-9)

    valid_mask = (total_reviews >= 63) & (pos_ratio >= 0.65) & \
                 (~df['is_hollow'].fillna(False)) & \
                 (~df['is_delisted'].fillna(False)) & \
                 (~df['is_utility'].fillna(False)) & \
                 (~df['is_nsfw'].fillna(False))
                 
    total_sim[target_idx] = -1
    total_sim[~valid_mask] = -1

    # Apply puzzle firewall
    for idx in np.where(valid_mask)[0]:
        if t_subgenre != 'Generic/Other':
            m_tags = get_list(df.loc[idx].get('tags', []))
            m_subgenre = identify_puzzle_subgenre(m_tags)
            if m_subgenre != 'Generic/Other' and m_subgenre != t_subgenre:
                total_sim[idx] -= 0.5 
                
    top_indices = np.argsort(total_sim)[::-1][:50]
    print(f"\nTarget: {t_row['name']} (Puzzle Subgenre: {t_subgenre})")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {df.loc[idx]['name']} (Sim: {total_sim[idx]:.3f})")

if __name__ == "__main__":
    evaluate_similarity("Detroit: Become Human")
    evaluate_similarity("Disco Elysium")
    evaluate_similarity("NieR:Automata")
    evaluate_similarity("The Talos Principle")
