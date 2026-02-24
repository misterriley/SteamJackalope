import numpy as np
import pandas as pd
import json
import os
import sys
from scipy.spatial.distance import jensenshannon

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import METADATA_FILE, ROOT_DIR

def compare_jsd(id1, id2):
    df = pd.read_parquet(METADATA_FILE)
    dist_path = os.path.join(ROOT_DIR, "data", "production", "topic_distributions.npy")
    desc_path = os.path.join(ROOT_DIR, "data", "production", "topic_descriptions.json")
    
    probs = np.load(dist_path, mmap_mode='r')
    with open(desc_path, 'r') as f:
        descriptions = json.load(f)
        
    idx1 = df[df.appid == id1].index[0]
    idx2 = df[df.appid == id2].index[0]
    
    p = probs[idx1].astype(np.float64)
    q = probs[idx2].astype(np.float64)
    
    # Calculate JS Distance
    distance = jensenshannon(p, q)
    similarity = 1.0 - distance
    
    print(f"--- Comparison: {df.iloc[idx1]['name']} vs {df.iloc[idx2]['name']} ---")
    print(f"JS Distance:   {distance:.4f}")
    print(f"JS Similarity: {similarity:.4f} (1.0 is identical)")
    
    diff = p - q
    top_diff_indices = np.argsort(np.abs(diff))[::-1][:10]
    
    print("\n--- Top Divergent Topics ---")
    print(f"{'Topic':<30} | {'Game A':<8} | {'Game B':<8} | {'Delta'}")
    print("-" * 65)
    for i in top_diff_indices:
        desc = descriptions.get(str(i), f"Topic {i}")
        print(f"{desc[:30]:<30} | {p[i]:>7.1%} | {q[i]:>7.1%} | {diff[i]:>+7.1%}")

if __name__ == "__main__":
    compare_jsd(400, 620) # Portal 1 vs 2
