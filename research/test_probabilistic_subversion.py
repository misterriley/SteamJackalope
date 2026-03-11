import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import Ridge

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

def calculate_subversion_probability(tags_set):
    """
    Returns a probability (0.0 to 1.0) that a game is a structural subversion 
    based on the interaction of specific tags.
    """
    prob = 0.0
    has_psych_horror = 'Psychological Horror' in tags_set
    has_surreal = 'Surreal' in tags_set
    has_satire = 'Satire' in tags_set
    
    innocent_tags = {'Cute', 'Education', 'Dating Sim', 'Family Friendly', 'Farming Sim', 'Typing', 'Math', 'Software', 'Game Development'}
    intersecting_innocent = tags_set.intersection(innocent_tags)
    
    if len(intersecting_innocent) > 0:
        # High Probability triggers
        if has_psych_horror:
            if 'Dating Sim' in intersecting_innocent:
                prob = max(prob, 0.95)
            elif 'Cute' in intersecting_innocent or 'Family Friendly' in intersecting_innocent:
                prob = max(prob, 0.85)
            else:
                prob = max(prob, 0.60)
        
        # Medium Probability triggers
        if has_satire:
            if 'Farming Sim' in intersecting_innocent or 'Game Development' in intersecting_innocent:
                prob = max(prob, 0.40)
            else:
                prob = max(prob, 0.25)
                
        # Low Probability triggers
        if has_surreal:
            # Surreal + Cute is often just a normal indie game, not a subversion
            if 'Education' in intersecting_innocent or 'Math' in intersecting_innocent:
                prob = max(prob, 0.30)
            else:
                prob = max(prob, 0.15)
                
    return prob

def main():
    print("Loading metadata...")
    df = pd.read_parquet('data/production/metadata.parquet')
    
    print("Finding SpaceChem and Anthology of the Killer...")
    # Find SpaceChem
    spacechem = df[df['name'] == 'SpaceChem']
    if len(spacechem) == 0:
        print("SpaceChem not found!")
        return
    idx_spacechem = spacechem.index[0]
    
    # Find Anthology of the Killer (3212530)
    anthology = df[df['appid'] == 3212530]
    if len(anthology) == 0:
        print("Anthology of the Killer not found!")
        return
    idx_anthology = anthology.index[0]
    
    print("Finding Undertale and Doki Doki Literature Club Plus!...")
    undertale = df[df['name'] == 'Undertale']
    idx_undertale = undertale.index[0] if len(undertale) > 0 else -1
    
    ddlc = df[df['name'] == 'Doki Doki Literature Club Plus!']
    idx_ddlc = ddlc.index[0] if len(ddlc) > 0 else -1
    
    print("\nLoading feature matrices...")
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))
    pop_z = df['pop_z'].fillna(0).values
    pop_discount = np.where(pop_z > 0, np.exp(-0.15 * pop_z), 1.0)
    
    def test_pair(name_a, idx_a, name_b, idx_b):
        if idx_a == -1 or idx_b == -1: return
        print(f"\n{'='*50}")
        print(f"Testing {name_a} vs {name_b}")
        print(f"{'='*50}")
        
        tags_a = set(get_list(df.iloc[idx_a]['tags']))
        tags_b = set(get_list(df.iloc[idx_b]['tags']))
        
        sim_tags = np.dot(f_tags[idx_a], f_tags[idx_b])
        sim_desc = np.dot(f_desc[idx_a], f_desc[idx_b])
        sim_verbs = np.dot(f_verbs[idx_a], f_verbs[idx_b])
        sim_graph_raw = np.dot(f_graph[idx_a], f_graph[idx_b])
        
        discount_b = pop_discount[idx_b]
        sim_graph = sim_graph_raw * discount_b
        
        total_base = (0.174 * sim_tags) + (0.445 * sim_desc) + (0.233 * sim_verbs) + (0.148 * sim_graph)
        print(f"Base Similarity: {total_base:.4f}")
        
        prob_a = calculate_subversion_probability(tags_a)
        prob_b = calculate_subversion_probability(tags_b)
        print(f"{name_a} Subversion Prob: {prob_a:.2f}")
        print(f"{name_b} Subversion Prob: {prob_b:.2f}")
        
        # New probabilistic scaling (Max theoretical bonus of +0.45, scaled by the product of probabilities)
        # We use sqrt to make it slightly more forgiving if one is 1.0 and the other is 0.6
        joint_prob = np.sqrt(prob_a * prob_b)
        subv_bonus = 0.45 * joint_prob
        print(f"Joint Subversion Probability: {joint_prob:.2f}")
        print(f"Calculated Bonus: +{subv_bonus:.4f}")
        
        final_sim = total_base + subv_bonus
        print(f"FINAL SIMILARITY: {final_sim:.4f}")

    test_pair('SpaceChem', idx_spacechem, 'Anthology of the Killer', idx_anthology)
    test_pair('Undertale', idx_undertale, 'Doki Doki Literature Club Plus!', idx_ddlc)
    test_pair('SpaceChem', idx_spacechem, 'Factorio', df[df['name'] == 'Factorio'].index[0])

if __name__ == "__main__":
    main()
