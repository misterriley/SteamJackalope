import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import TOPIC_DISTRIBUTIONS_FILE, METADATA_FILE

def match_motivation_destruction():
    print("Loading topic distributions...")
    probs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    
    # Topic 210 is "destruction" related
    target_topic_idx = 210
    
    print(f"Loading metadata to get game names for Topic {target_topic_idx}...")
    df = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    
    # Get loadings for topic 210
    loadings = probs[:, target_topic_idx]
    
    # Get top 10 indices
    top_10_indices = np.argsort(-loadings)[:10]
    
    print(f"\nTop 10 Games for Topic {target_topic_idx} (Destruction/Kaiju):")
    for i, idx in enumerate(top_10_indices):
        game = df.iloc[idx]
        print(f"{i+1}. {game['name']} (AppID: {game['appid']}, Loading: {loadings[idx]:.4f})")

if __name__ == "__main__":
    match_motivation_destruction()
