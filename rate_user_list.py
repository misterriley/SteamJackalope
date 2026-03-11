import pandas as pd
import numpy as np

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

def identify_puzzle_subgenre(tags_set):
    if 'Hidden Object' in tags_set: return 'Hidden Object'
    if 'Automation' in tags_set or 'Programming' in tags_set: return 'Automation'
    if 'Sokoban' in tags_set or 'Grid-Based Movement' in tags_set: return 'Sokoban/Grid'
    if 'Puzzle' in tags_set and ('First-Person' in tags_set or '3D Platformer' in tags_set or 'Open World' in tags_set): return 'Spatial/3D'
    return 'Generic/Other'

def calculate_subversion_score(tags_set):
    meta_tags = {'Psychological Horror', 'Fourth Wall', 'Surreal', 'Satire', 'Parody', 'Illuminati', 'Mind-Bending'}
    innocent_tags = {'Cute', 'Education', 'Dating Sim', 'Family Friendly', 'Farming Sim', 'Typing', 'Math', 'Software', 'Game Development'}
    
    meta_count = len(tags_set.intersection(meta_tags))
    innocent_count = len(tags_set.intersection(innocent_tags))
    
    if meta_count >= 1 and innocent_count >= 1: return 3.0
    elif meta_count >= 2: return 2.0
    elif meta_count == 1: return 1.0
    return 0.0

def main():
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    features = {}
    features['tags'] = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    features['desc'] = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    features['verbs'] = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    features['graph'] = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))

    gt = pd.read_csv('data/user_76561198039155404_ground_truth.csv')
    rated_appids = set(gt[gt['status'] == 'rated']['appid'].tolist())
    print(f"Found {len(rated_appids)} rated games in user list.")

    target_appid = 1194840 # Frog Fractions
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}

    matches = df[df['appid'] == target_appid]
    if len(matches) == 0:
        print("Target game not found in metadata.")
        return

    target_idx = matches.index[0]
    t_row = df.loc[target_idx]
    t_tags = set(get_list(t_row.get('tags', [])))
    t_subgenre = identify_puzzle_subgenre(t_tags)
    t_subv = calculate_subversion_score(t_tags)

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

    # We won't apply quality_penalty here because we only care about scoring the user's rated list.
    # The user's list represents ground truth games, so they bypass shovelware filters.
    
    # We DO apply the puzzle firewall and subversion matching
    for idx in range(len(total_sim)):
        m_appid = df.loc[idx, 'appid']
        if m_appid not in rated_appids:
            total_sim[idx] = -999.0 # Exclude games not in the user's rated list
            continue
            
        if idx == target_idx:
            total_sim[idx] = -999.0 # Exclude the target game itself
            continue

        m_tags = set(get_list(df.loc[idx].get('tags', [])))
        
        # Puzzle Firewall
        if t_subgenre != 'Generic/Other':
            m_subgenre = identify_puzzle_subgenre(m_tags)
            if m_subgenre != 'Generic/Other' and m_subgenre != t_subgenre:
                total_sim[idx] -= 0.3 
                
        # Subversion Matching
        m_subv = calculate_subversion_score(m_tags)
        if t_subv >= 3.0:
            if m_subv >= 3.0: total_sim[idx] += 0.45
            elif m_subv >= 2.0: total_sim[idx] += 0.25
            else: total_sim[idx] -= 0.30
        elif t_subv >= 2.0:
            if m_subv >= 2.0: total_sim[idx] += 0.25
            else: total_sim[idx] -= 0.20
        elif t_subv == 0.0:
            if m_subv >= 2.0: total_sim[idx] -= 0.30

    # Get indices for the user's rated games, sorted by score
    valid_indices = np.where(total_sim > -900.0)[0]
    sorted_indices = valid_indices[np.argsort(total_sim[valid_indices])[::-1]]
    
    print(f"\n{'='*50}\nTarget: {t_row['name']} (Subversion Score: {t_subv})\n{'='*50}")
    print(f"Ranking the {len(sorted_indices)} other rated games in user's list:\n")
    
    for i, idx in enumerate(sorted_indices):
        m_row = df.loc[idx]
        m_tags = set(get_list(m_row.get('tags', [])))
        m_subv = calculate_subversion_score(m_tags)
        user_rating = gt[gt['appid'] == m_row['appid']]['actual_rating'].values[0]
        
        # Print top 25 and bottom 10
        if i < 25 or i >= len(sorted_indices) - 10:
            print(f"{i+1:3d}. {m_row['name'][:40]:<40} | Sim: {total_sim[idx]:.3f} | Subv: {m_subv} | User Rating: {user_rating}")
        elif i == 25:
            print("...")

if __name__ == "__main__":
    main()
