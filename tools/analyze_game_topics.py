
import pandas as pd
import numpy as np
import pickle
import os
import sys
import argparse
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import METADATA_FILE, ROOT_DIR

def analyze_game_topics(appid):
    # 1. Load Data
    df = pd.read_parquet(METADATA_FILE)
    dist_path = os.path.join(ROOT_DIR, "data", "production", "topic_distributions.npy")
    model_path = os.path.join(ROOT_DIR, "data", "production", "topic_model.pkl")
    
    if not os.path.exists(dist_path) or not os.path.exists(model_path):
        print("Required artifacts not found.")
        return
        
    probs = np.load(dist_path, mmap_mode='r')
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
    
    # 2. Find Game Index
    match = df[df.appid == appid]
    if match.empty:
        print(f"AppID {appid} not found.")
        return
        
    idx = match.index[0]
    name = match.iloc[0]['name']
    
    # 3. Get Loadings
    # The columns in topic_distributions.npy correspond to the ORDER of topics in topic_model.get_topic_info()
    topic_info = topic_model.get_topic_info()
    # Filter out -1 (outliers) as it's not in the distribution matrix
    valid_topics = topic_info[topic_info.Topic != -1]
    
    row = probs[idx].astype(np.float32)
    top_indices = np.argsort(-row)[:10]
    
    header = f"\n--- Top Topic Loadings: {name} (ID: {appid}) ---"
    print(header)
    print(f"{'Topic ID':<10} | {'Probability':<10} | {'Top Words'}")
    print("-" * 75)
    
    for i in top_indices:
        p = row[i]
        if p > 0.001: 
            # i is the column index, we need to find which Topic ID that corresponds to
            topic_id = valid_topics.iloc[i]['Topic']
            words = topic_model.get_topic(topic_id)
            word_str = ", ".join([w[0] for w in words[:5]])
            print(f"{topic_id:<10} | {p:<10.2%} | {word_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("appid", type=int, help="Steam AppID to analyze")
    args = parser.parse_args()
    
    analyze_game_topics(args.appid)
