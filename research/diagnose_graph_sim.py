import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, PRODUCTION_DATA_DIR

def diagnose_graph():
    meta = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): i for i, aid in enumerate(meta['appid'])}
    seed_idx = appid_to_idx[1794680] # Vampire Survivors
    
    g_vecs = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r')
    seed_g = g_vecs[seed_idx]
    
    g_dot = np.dot(g_vecs, seed_g)
    g_norms = np.linalg.norm(g_vecs, axis=1) * np.linalg.norm(seed_g)
    g_sims = g_dot / (g_norms + 1e-9)
    
    # Raw top 10 (excluding self)
    top_10_idx = np.argsort(-g_sims)[1:11]
    
    print("\n--- TOP 10 RAW GRAPH NEIGHBORS (NO TASTE BIAS) ---")
    for idx in top_10_idx:
        print(f"{meta.iloc[idx]['name']} (Sim: {g_sims[idx]:.3f})")
        
    # Check Halls of Torment
    h_idx = appid_to_idx.get(2218750)
    if h_idx is not None:
        print(f"\nSpecific Check: Halls of Torment Graph Sim: {g_sims[h_idx]:.3f} (Rank: {np.where(np.argsort(-g_sims) == h_idx)[0][0]})")

if __name__ == '__main__':
    diagnose_graph()
