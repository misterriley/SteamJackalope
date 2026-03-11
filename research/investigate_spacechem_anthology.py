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

def main():
    print("Loading metadata...")
    df = pd.read_parquet('data/production/metadata.parquet')
    
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
    
    print(f"\nSpaceChem (AppID: {spacechem['appid'].values[0]}) vs Anthology of the Killer (AppID: {anthology['appid'].values[0]})")
    
    print("\n--- Game Profiles ---")
    print(f"SpaceChem Tags: {get_list(spacechem['tags'].values[0])}")
    print(f"Anthology Tags: {get_list(anthology['tags'].values[0])}")
    print(f"\nSpaceChem Desc: {spacechem['short_description'].values[0][:200]}...")
    print(f"Anthology Desc: {anthology['short_description'].values[0][:200]}...")
    
    print("\nLoading feature matrices...")
    f_tags = normalize(np.load('data/production/steam_tag_vectors.npy', mmap_mode='r'))
    f_desc = normalize(np.load('data/production/embeddings_desc.npy', mmap_mode='r'))
    f_verbs = normalize(np.load('data/production/diffused_verb_profiles.npy', mmap_mode='r').astype(np.float32))
    f_graph = normalize(np.load('data/production/embeddings_graph.npy', mmap_mode='r'))
    
    sim_tags = np.dot(f_tags[idx_spacechem], f_tags[idx_anthology])
    sim_desc = np.dot(f_desc[idx_spacechem], f_desc[idx_anthology])
    sim_verbs = np.dot(f_verbs[idx_spacechem], f_verbs[idx_anthology])
    
    pop_z_anthology = anthology['pop_z'].values[0]
    pop_discount = np.exp(-0.15 * pop_z_anthology) if pop_z_anthology > 0 else 1.0
    sim_graph_raw = np.dot(f_graph[idx_spacechem], f_graph[idx_anthology])
    sim_graph_discounted = sim_graph_raw * pop_discount
    
    print("\n--- Similarity Breakdown ---")
    print(f"Tags Sim  (weight 0.174): {sim_tags:.4f}")
    print(f"Desc Sim  (weight 0.445): {sim_desc:.4f}")
    print(f"Verbs Sim (weight 0.233): {sim_verbs:.4f}")
    print(f"Graph Sim (weight 0.148): {sim_graph_raw:.4f} (discounted: {sim_graph_discounted:.4f})")
    
    total_sim = (0.174 * sim_tags) + (0.445 * sim_desc) + (0.233 * sim_verbs) + (0.148 * sim_graph_discounted)
    print(f"\nTotal Base Similarity: {total_sim:.4f}")
    
    # Subgenre firewall check
    def identify_puzzle_subgenre(tags_set):
        if 'Hidden Object' in tags_set: return 'Hidden Object'
        if 'Automation' in tags_set or 'Programming' in tags_set: return 'Automation'
        if 'Sokoban' in tags_set or 'Grid-Based Movement' in tags_set: return 'Sokoban/Grid'
        if 'Puzzle' in tags_set and ('First-Person' in tags_set or '3D Platformer' in tags_set or 'Open World' in tags_set): return 'Spatial/3D'
        return 'Generic/Other'

    sc_subg = identify_puzzle_subgenre(set(get_list(spacechem['tags'].values[0])))
    an_subg = identify_puzzle_subgenre(set(get_list(anthology['tags'].values[0])))
    print(f"\nSpaceChem Puzzle Subgenre: {sc_subg}")
    print(f"Anthology Puzzle Subgenre: {an_subg}")
    if sc_subg != 'Generic/Other' and an_subg != 'Generic/Other' and sc_subg != an_subg:
         print("-> Puzzle Firewall APPLIED: -0.3 penalty")
         total_sim -= 0.3
         
    # Subversion check
    def calculate_subversion_score(tags_set):
        meta_tags = {'Psychological Horror', 'Fourth Wall', 'Surreal', 'Satire', 'Parody', 'Illuminati', 'Mind-Bending'}
        innocent_tags = {'Cute', 'Education', 'Dating Sim', 'Family Friendly', 'Farming Sim', 'Typing', 'Math', 'Software', 'Game Development'}
        meta_count = len(tags_set.intersection(meta_tags))
        innocent_count = len(tags_set.intersection(innocent_tags))
        if meta_count >= 1 and innocent_count >= 1: return 3.0
        elif meta_count >= 2: return 2.0
        elif meta_count == 1: return 1.0
        return 0.0

    sc_subv = calculate_subversion_score(set(get_list(spacechem['tags'].values[0])))
    an_subv = calculate_subversion_score(set(get_list(anthology['tags'].values[0])))
    print(f"SpaceChem Subversion Tier: {sc_subv}")
    print(f"Anthology Subversion Tier: {an_subv}")
    if sc_subv == 0.0 and an_subv >= 2.0:
        print("-> Target has no subversion, match has Tier 2+: -0.3 penalty")
        total_sim -= 0.3
    
    print(f"\nFINAL ADJUSTED SIMILARITY: {total_sim:.4f}")

if __name__ == "__main__":
    main()
