import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import METADATA_FILE, ROOT_DIR

def fast_jsd(P, Q_matrix):
    """
    Calculates JS Distance between a single distribution P 
    and a matrix of distributions Q in a vectorized way.
    """
    # Add epsilon to avoid log(0)
    eps = 1e-10
    P = P + eps
    Q_matrix = Q_matrix + eps
    
    M = 0.5 * (P + Q_matrix)
    
    # Vectorized KL Divergence
    def kld(X, Y):
        return np.sum(X * np.log(X / Y), axis=-1)
    
    js_div = 0.5 * kld(P, M) + 0.5 * kld(Q_matrix, M)
    
    # Clip tiny negative values from precision errors before sqrt
    return np.sqrt(np.maximum(js_div, 0))

def search_similar_jsd(appid, top_k=10):
    # 1. Load Data
    print("Loading datasets...")
    df = pd.read_parquet(METADATA_FILE)
    dist_path = os.path.join(ROOT_DIR, "data", "production", "topic_distributions.npy")
    desc_path = os.path.join(ROOT_DIR, "data", "production", "topic_descriptions.json")
    
    probs = np.load(dist_path).astype(np.float32)
    with open(desc_path, 'r') as f:
        descriptions = json.load(f)
        
    match = df[df.appid == appid]
    if match.empty:
        print(f"AppID {appid} not found.")
        return
        
    query_idx = match.index[0]
    query_name = match.iloc[0]['name']
    query_p = probs[query_idx]
    
    print(f"Searching for neighbors of '{query_name}' using JS Divergence...")
    
    # 2. Calculate Distances (Vectorized)
    distances = fast_jsd(query_p, probs)
    
    # 3. Filter Query Game
    distances[query_idx] = 1e12
    
    # 4. Get Top K
    top_indices = np.argsort(distances)[:top_k]
    
    print("\n--- Top " + str(top_k) + " JS Similarity Matches for " + query_name + " ---")
    print(f"{'Rank':<4} | {'AppID':<10} | {'Similarity':<10} | {'Name'}")
    print("-" * 70)
    
    for i, idx in enumerate(top_indices):
        sim = 1.0 - distances[idx]
        name = df.iloc[idx]['name']
        aid = df.iloc[idx]['appid']
        print(f"{i+1:<4} | {aid:<10} | {sim:<10.4f} | {name}")
        
    # Show primary topics for the #1 match for debug
    best_idx = top_indices[0]
    best_p = probs[best_idx]
    best_name = df.iloc[best_idx]['name']
    
    print("\n--- Primary Topic Comparison (Query vs Rank #1: " + best_name + ") ---")
    top_q = np.argsort(-query_p)[:5]
    for t in top_q:
        if query_p[t] > 0.01:
            desc = descriptions.get(str(t), f"Topic {t}")
            print(f"Topic {t:<3} ({desc[:25]:<25}): {query_p[t]:>6.1%} vs {best_p[t]:>6.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("appid", type=int, help="Steam AppID to search from")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    args = parser.parse_args()
    
    search_similar_jsd(args.appid, args.k)
