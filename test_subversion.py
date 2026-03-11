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
    # Core meta tags that indicate a game messes with the player or genre
    meta_tags = {'Psychological Horror', 'Fourth Wall', 'Surreal', 'Satire', 'Parody', 'Illuminati', 'Mind-Bending'}
    
    # Innocent or purely mechanical tags that usually don't mix with meta tags
    innocent_tags = {'Cute', 'Education', 'Dating Sim', 'Family Friendly', 'Farming Sim', 'Typing', 'Math', 'Software', 'Game Development'}
    
    meta_count = len(tags_set.intersection(meta_tags))
    innocent_count = len(tags_set.intersection(innocent_tags))
    
    # Tier 3: The Ultimate Subversion (e.g. DDLC, Frog Fractions, Pony Island)
    if meta_count >= 1 and innocent_count >= 1:
        return 3.0
    # Tier 2: Heavily Meta/Surreal games (e.g. Stanley Parable, Antichamber)
    elif meta_count >= 2:
        return 2.0
    # Tier 1: Mildly meta or just dark
    elif meta_count == 1:
        return 1.0
    
    return 0.0

def evaluate_similarity(target_name, df, features):
    weights = {'tags': 0.174, 'desc': 0.445, 'verbs': 0.233, 'graph': 0.148}

    matches = df[df['name'] == target_name]
    if len(matches) == 0: matches = df[df['name'].str.contains(target_name, case=False, na=False)]
    if len(matches) == 0:
        print(f"Game '{target_name}' not found.")
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

    total_reviews = df['positive'].fillna(0) + df['negative'].fillna(0)
    pos_ratio = df['positive'].fillna(0) / (total_reviews + 1e-9)

    quality_penalty = np.where(
        (total_reviews < 63) | 
        (pos_ratio < 0.65) | 
        (df['is_hollow'].fillna(False)) | 
        (df['is_delisted'].fillna(False)) | 
        (df['is_utility'].fillna(False)) | 
        (df['is_nsfw'].fillna(False)),
        -0.4, 0.0
    )
    total_sim += quality_penalty
    total_sim[target_idx] = -2.0

    for idx in range(len(total_sim)):
        m_tags = set(get_list(df.loc[idx].get('tags', [])))
        
        if t_subgenre != 'Generic/Other':
            m_subgenre = identify_puzzle_subgenre(m_tags)
            if m_subgenre != 'Generic/Other' and m_subgenre != t_subgenre:
                total_sim[idx] -= 0.3 
                
        m_subv = calculate_subversion_score(m_tags)
        
        # Cross-genre Subversion Bridging
        # If the target is highly subversive (Tier 3), it wants to find other Tier 3/2 games, regardless of genre
        if t_subv >= 3.0:
            if m_subv >= 3.0:
                total_sim[idx] += 0.45  # Massive magnet for other genre-subversions
            elif m_subv >= 2.0:
                total_sim[idx] += 0.25
            else:
                total_sim[idx] -= 0.30  # Penalty for playing it straight
        elif t_subv >= 2.0:
            if m_subv >= 2.0:
                total_sim[idx] += 0.25
            else:
                total_sim[idx] -= 0.20
        # If the target is NOT subversive, do not recommend subversive games
        elif t_subv == 0.0:
            if m_subv >= 2.0:
                total_sim[idx] -= 0.30
                
    top_indices = np.argsort(total_sim)[::-1][:20]
    print(f"\n{'='*50}\nTarget: {t_row['name']} (Subversion Score: {t_subv})\n{'='*50}")
    for i, idx in enumerate(top_indices):
        m_tags = set(get_list(df.loc[idx].get('tags', [])))
        m_subv = calculate_subversion_score(m_tags)
        print(f"  {i+1}. {df.loc[idx]['name']} (Sim: {total_sim[idx]:.3f}) | Subv: {m_subv}")

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet('data/production/metadata.parquet')
    features = {}
    features['tags'] = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    features['desc'] = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    features['verbs'] = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    features['graph'] = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))
    
    evaluate_similarity("Frog Fractions: Game of the Decade Edition", df, features)
    evaluate_similarity("Doki Doki Literature Club!", df, features)
    evaluate_similarity("MathLand", df, features)
