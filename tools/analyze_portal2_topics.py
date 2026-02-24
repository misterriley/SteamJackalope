
import pandas as pd
import numpy as np
import json
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, ROOT_DIR

def analyze_game_topics(appid=620):
    # 1. Load Data
    df = pd.read_parquet(METADATA_FILE)
    dist_path = os.path.join(ROOT_DIR, "data", "production", "topic_distributions.npy")
    desc_path = os.path.join(ROOT_DIR, "data", "production", "topic_descriptions.json")
    
    if not os.path.exists(dist_path):
        print("Topic distributions not found.")
        return
        
    probs = np.load(dist_path, mmap_mode='r')
    with open(desc_path, 'r') as f:
        descriptions = json.load(f)
        
    # 2. Find Game Index
    match = df[df.appid == appid]
    if match.empty:
        print(f"AppID {appid} not found.")
        return
        
    idx = match.index[0]
    name = match.iloc[0]['name']
    
    # 3. Get Loadings
    row = probs[idx].astype(np.float32)
    top_indices = np.argsort(-row)[:10]
    
    print(f"--- Top 10 Topic Loadings: {name} ---")
    print(f"{'Topic ID':<10} | {'Probability':<10} | {'Expert Description'}")
    print("-" * 60)
    
    for i in top_indices:
        p = row[i]
        if p > 0:
            desc = descriptions.get(str(i), "Unknown Topic")
            print(f"{i:<10} | {p:<10.2%} | {desc}")

if __name__ == "__main__":
    analyze_game_topics(620) # Portal 2
