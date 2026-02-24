import pandas as pd
import numpy as np
import os
import sys
import time
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import TOPIC_DISTRIBUTIONS_FILE, ROOT_DIR

def fast_jsd_similarity_batch(P, Q):
    eps = 1e-10
    P = P + eps
    Q = Q + eps
    M = 0.5 * (P + Q)
    js_div = 0.5 * (np.sum(P * np.log(P / M), axis=-1) + np.sum(Q * np.log(Q / M), axis=-1))
    return 1.0 - np.sqrt(np.maximum(js_div, 0))

def calculate_stats(sample_size=5000):
    if not os.path.exists(TOPIC_DISTRIBUTIONS_FILE):
        print("Error: Topic distributions file not found.")
        return

    print("Loading topic distributions for " + str(sample_size) + " random comparisons...")
    data = np.load(TOPIC_DISTRIBUTIONS_FILE).astype(np.float32)
    num_games = data.shape[0]
    
    idx1 = np.random.choice(num_games, sample_size)
    idx2 = np.random.choice(num_games, sample_size)
    
    P = data[idx1]
    Q = data[idx2]
    
    print("Calculating JSD similarities...")
    start_time = time.time()
    similarities = fast_jsd_similarity_batch(P, Q)
    duration = time.time() - start_time
    
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    print("\n--- Topic Similarity Population Stats ---")
    print("Mean Similarity: " + f"{mean_sim:.6f}")
    print("Std Deviation:   " + f"{std_sim:.6f}")
    print("Min:             " + f"{np.min(similarities):.6f}")
    print("Max:             " + f"{np.max(similarities):.6f}")
    print("Calculation Time: " + f"{duration:.2f}s")
    
    stats = {
        "TOPIC_SIMILARITY_MEAN": float(mean_sim),
        "TOPIC_SIMILARITY_STD": float(std_sim)
    }
    
    with open(os.path.join(ROOT_DIR, "data", "production", "topic_similarity_stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)
    print("\nStats saved to data/production/topic_similarity_stats.json")

if __name__ == "__main__":
    calculate_stats(10000)
