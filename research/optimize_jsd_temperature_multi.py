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
    js_div = 0.5 * (np.sum(p * np.log(p / m), axis=-1) + np.sum(Q * np.log(Q / m), axis=-1))
    return np.sqrt(np.maximum(js_div, 0))

def run_multi_game_sweep():
    games = [
        {"id": 1222140, "name": "Detroit: Become Human"},
        {"id": 292030, "name": "The Witcher 3"},
        {"id": 1245620, "name": "Elden Ring"},
        {"id": 413150, "name": "Stardew Valley"}
    ]
    
    # 1. Load Data
    df = pd.read_parquet(METADATA_FILE)
    model_path = os.path.join(ROOT_DIR, "data", "production", "topic_model.pkl")
    embeddings = np.load(EMBEDDINGS_DESC_RAW_FILE).astype(np.float32)
    
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
        
    topic_embeddings = topic_model.topic_embeddings_[1:] 
    sim_matrix = cosine_similarity(embeddings, topic_embeddings)
    sim_matrix = sim_matrix - np.max(sim_matrix, axis=1, keepdims=True)
    
    temperatures = [0.01, 0.05, 0.1]
    
    for game in games:
        query_appid = game['id']
        match = df[df.appid == query_appid]
        if match.empty:
            print("\nSkipping " + game['name'] + " (ID " + str(query_appid) + " not found)")
            continue
            
        query_idx = match.index[0]
        print("\n==================================================")
        print("GAME: " + game['name'] + " (ID: " + str(query_appid) + ")")
        print("==================================================")
        
        for T in temperatures:
            print("\nEvaluating T=" + str(T) + "...")
            exp_sim = np.exp(sim_matrix / T)
            probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
            
            query_p = probs[query_idx]
            distances = fast_jsd_single(query_p, probs)
            distances[query_idx] = 1e12 
            
            top_indices = np.argsort(distances)[:5]
            
            for i, idx in enumerate(top_indices):
                sim = 1.0 - distances[idx]
                name = df.iloc[idx]['name']
                print("  " + str(i+1) + ". " + name + " (" + f"{sim:.4f}" + ")")

if __name__ == "__main__":
    run_multi_game_sweep()
