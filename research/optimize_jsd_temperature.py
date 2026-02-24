import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import METADATA_FILE, ROOT_DIR, EMBEDDINGS_DESC_RAW_FILE

def fast_jsd_single(p, Q_matrix):
    eps = 1e-10
    p = p + eps
    Q = Q_matrix + eps
    m = 0.5 * (p + Q)
    # JS Div = 0.5 * KLD(P||M) + 0.5 * KLD(Q||M)
    js_div = 0.5 * (np.sum(p * np.log(p / m), axis=-1) + np.sum(Q * np.log(Q / m), axis=-1))
    return np.sqrt(np.maximum(js_div, 0))

def run_temperature_sweep(query_appid=620):
    print(f"--- JSD Temperature Sweep: {query_appid} ---")
    
    # 1. Load Data
    df = pd.read_parquet(METADATA_FILE)
    model_path = os.path.join(ROOT_DIR, "data", "production", "topic_model.pkl")
    embeddings = np.load(EMBEDDINGS_DESC_RAW_FILE).astype(np.float32)
    
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
        
    query_idx = df[df.appid == query_appid].index[0]
    query_name = df.iloc[query_idx]['name']
    
    # 2. Get Topic Centroids (Aligned)
    topic_embeddings = topic_model.topic_embeddings_[1:] # Skip outliers
    
    # 3. Pre-calculate Cosine Similarity Matrix
    print("Calculating base similarity matrix...")
    sim_matrix = cosine_similarity(embeddings, topic_embeddings)
    sim_matrix = sim_matrix - np.max(sim_matrix, axis=1, keepdims=True) # Stability
    
    # 4. Sweep Temperatures
    temperatures = [0.01, 0.025, 0.05, 0.1, 0.2]
    
    for T in temperatures:
        print("\nEvaluating T=" + str(T) + "...")
        
        # Softmax
        exp_sim = np.exp(sim_matrix / T)
        probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        
        # JSD Search
        query_p = probs[query_idx]
        distances = fast_jsd_single(query_p, probs)
        distances[query_idx] = 1e12 # Ignore self
        
        top_indices = np.argsort(distances)[:5]
        
        print(f"{'Rank':<4} | {'Similarity':<10} | {'Game Name'}")
        print("-" * 50)
        for i, idx in enumerate(top_indices):
            sim = 1.0 - distances[idx]
            name = df.iloc[idx]['name']
            print(f"{i+1:<4} | {sim:<10.4f} | {name}")

if __name__ == "__main__":
    run_temperature_sweep(620) # Portal 2
