import pandas as pd
import numpy as np
import os

def test_clr_cosine():
    v = np.load('data/production/tag_prior_transformed.npy', mmap_mode='r')
    full_metadata = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name'])
    
    idx_o = full_metadata[full_metadata['appid']==1057090].index[0]
    idx_h = full_metadata[full_metadata['appid']==367520].index[0]
    idx_w = full_metadata[full_metadata['appid']==245450].index[0]
    idx_g = full_metadata[full_metadata['appid']==534550].index[0] # Guacamelee 2
    
    def cos(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    
    v_o = v[idx_o]
    v_h = v[idx_h]
    v_w = v[idx_w]
    v_g = v[idx_g]
    
    print("CLR (Non-whitened) Cosine Similarity to Ori:")
    print(f"Hollow Knight: {cos(v_o, v_h):.4f}")
    print(f"Guacamelee 2:  {cos(v_o, v_g):.4f}")
    print(f"Wizardry 8:    {cos(v_o, v_w):.4f}")

if __name__ == "__main__":
    test_clr_cosine()
