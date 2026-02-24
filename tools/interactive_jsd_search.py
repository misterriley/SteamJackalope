import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import METADATA_FILE, ROOT_DIR, MODEL_NAME

def fast_jsd_single(p, Q_matrix):
    eps = 1e-10
    p = p + eps
    Q = Q_matrix + eps
    m = 0.5 * (p + Q)
    js_div = 0.5 * (np.sum(p * np.log(p / m), axis=-1) + np.sum(Q * np.log(Q / m), axis=-1))
    return np.sqrt(np.maximum(js_div, 0))

def interactive_search():
    print("--- SteamJackalope: Interactive JSD Topic Search ---")
    print("Loading resources (this may take 15-20 seconds)...")
    
    # 1. Load Data
    df = pd.read_parquet(METADATA_FILE)
    dist_path = os.path.join(ROOT_DIR, "data", "production", "topic_distributions.npy")
    model_path = os.path.join(ROOT_DIR, "data", "production", "topic_model.pkl")
    desc_path = os.path.join(ROOT_DIR, "data", "production", "topic_descriptions.json")
    
    if not os.path.exists(dist_path) or not os.path.exists(model_path):
        print("Error: Production artifacts missing. Please run pipeline/generate_topic_model.py first.")
        return

    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
        
    probs_matrix = np.load(dist_path).astype(np.float32)
    
    descriptions = {}
    if os.path.exists(desc_path):
        with open(desc_path, 'r') as f:
            descriptions = json.load(f)
            
    # 2. Load Model
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Setup Loop
    T = 0.05
    top_k = 10
    topic_embeddings = topic_model.topic_embeddings_[1:] # Aligned
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info.Topic != -1]
    
    print("\nReady! Commands:")
    print("  - Just type a prompt to search.")
    print("  - '/temp <val>' to change temperature (current: " + str(T) + ")")
    print("  - '/exit' to quit.")
    
    while True:
        try:
            user_input = input("\nPrompt > ").strip()
        except EOFError:
            break
            
        if not user_input:
            continue
        if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
            break
        if user_input.startswith('/temp'):
            try:
                T = float(user_input.split()[1])
                print("Temperature updated to: " + str(T))
            except:
                print("Invalid temperature format. Use '/temp 0.05'")
            continue

        # Process Prompt
        prompt_vec = model.encode([user_input.lower()])[0].astype(np.float32)
        sims = cosine_similarity(prompt_vec.reshape(1, -1), topic_embeddings)[0]
        
        # Softmax
        sims = sims - np.max(sims)
        exp_sim = np.exp(sims / T)
        prompt_p = exp_sim / np.sum(exp_sim)
        
        # Loadings
        print("\n[Topic Analysis]")
        top_t_indices = np.argsort(-prompt_p)[:5]
        for i in top_t_indices:
            p = prompt_p[i]
            if p > 0.005:
                tid = valid_topics.iloc[i]['Topic']
                desc = descriptions.get(str(tid), ", ".join([w[0] for w in topic_model.get_topic(tid)[:3]]))
                print("  - " + desc + " (" + f"{p:>5.1%}" + ")")
                
        # JSD Search
        distances = fast_jsd_single(prompt_p, probs_matrix)
        top_indices = np.argsort(distances)[:top_k]
        
        print("\n[Top Matches]")
        for i, idx in enumerate(top_indices):
            sim = 1.0 - distances[idx]
            name = df.iloc[idx]['name']
            print("  " + str(i+1) + ". " + name + " (" + f"{sim:.4f}" + ")")

if __name__ == "__main__":
    interactive_search()
