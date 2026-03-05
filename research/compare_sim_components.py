import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, PRODUCTION_DATA_DIR, TOPIC_DISTRIBUTIONS_FILE

def compare_components():
    meta = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): i for i, aid in enumerate(meta['appid'])}
    
    v_idx = appid_to_idx[1794680] # Vampire Survivors
    o_idx = appid_to_idx[2133330] # The Otter Ways
    h_idx = appid_to_idx[2218750] # Halls of Torment
    
    g_vecs = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r')
    t_dists = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    
    def get_sims(idx1, idx2):
        g_sim = np.dot(g_vecs[idx1], g_vecs[idx2]) / (np.linalg.norm(g_vecs[idx1]) * np.linalg.norm(g_vecs[idx2]) + 1e-9)
        t_sim = np.dot(t_dists[idx1], t_dists[idx2])
        return g_sim, t_sim
    
    g_v_o, t_v_o = get_sims(v_idx, o_idx)
    g_v_h, t_v_h = get_sims(v_idx, h_idx)
    
    pop_v = meta.iloc[v_idx]['pop_z']
    pop_o = meta.iloc[o_idx]['pop_z']
    pop_h = meta.iloc[h_idx]['pop_z']
    
    print(f"\n--- Component Comparison vs Vampire Survivors ---")
    print(f"Halls of Torment: Graph={g_v_h:.3f}, Topic={t_v_h:.3f}, PopZ={pop_h:.2f}")
    print(f"The Otter Ways:   Graph={g_v_o:.3f}, Topic={t_v_o:.3f}, PopZ={pop_o:.2f}")

if __name__ == '__main__':
    compare_components()
