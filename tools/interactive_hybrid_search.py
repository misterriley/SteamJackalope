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
from common.constants import METADATA_FILE, ROOT_DIR, MODEL_NAME, EMBEDDINGS_DESC_RAW_FILE

def fast_jsd_single(p, Q_matrix):
    eps = 1e-10
    p = p + eps
    Q = Q_matrix + eps
    m = 0.5 * (p + Q)
    js_div = 0.5 * (np.sum(p * np.log(p / m), axis=-1) + np.sum(Q * np.log(Q / m), axis=-1))
    return np.sqrt(np.maximum(js_div, 0))

def interactive_search():
    print("--- SteamJackalope: Hybrid Vibe vs Topic Search ---")
    print("Loading resources...")
    
    # 1. Load Data
    df = pd.read_parquet(METADATA_FILE)
    dist_path = os.path.join(ROOT_DIR, "data", "production", "topic_distributions.npy")
    model_path = os.path.join(ROOT_DIR, "data", "production", "topic_model.pkl")
    raw_emb_path = EMBEDDINGS_DESC_RAW_FILE
    
    if not os.path.exists(dist_path) or not os.path.exists(model_path):
        print("Required artifacts missing. Run pipeline first.")
        return

    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
        
    probs_matrix = np.load(dist_path).astype(np.float32)
    raw_embeddings = np.load(raw_emb_path).astype(np.float32)
    
    # 2. Load ST Model
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Setup
    T = 0.05
    topic_embeddings = topic_model.topic_embeddings_[1:] 
    
    print("\nReady! Enter a prompt to compare Pure Vibe (ST) vs Topic Profile (JSD).")
    
    while True:
        try:
            user_input = input("\nPrompt > ").strip()
        except EOFError:
            break
            
        if user_input.lower() in ['/exit', 'exit', 'quit']:
            break
        if not user_input:
            continue

        # Process Prompt
        prompt_vec = model.encode([user_input.lower()])[0].astype(np.float32)
        
        # A. Pure Vibe (Cosine Similarity)
        vibe_sims = cosine_similarity(prompt_vec.reshape(1, -1), raw_embeddings)[0]
        vibe_top = np.argsort(-vibe_sims)[:10]
        
        # B. Topic JSD
        topic_sims_raw = cosine_similarity(prompt_vec.reshape(1, -1), topic_embeddings)[0]
        topic_sims_raw = topic_sims_raw - np.max(topic_sims_raw)
        prompt_p = np.exp(topic_sims_raw / T)
        prompt_p /= np.sum(prompt_p)
        
        jsd_distances = fast_jsd_single(prompt_p, probs_matrix)
        jsd_top = np.argsort(jsd_distances)[:10]
        
        print("\n" + "="*40)
        print("RESULTS FOR: " + user_input)
        print("="*40)
        
        print("\n[PURE VIBE: ST Cosine]           | [TOPIC PROFILE: JSD (T=0.05)]")
        print("-" * 85)
        for i in range(10):
            v_idx = vibe_top[i]
            j_idx = jsd_top[i]
            v_name = (df.iloc[v_idx]['name'][:30] + "...") if len(df.iloc[v_idx]['name']) > 30 else df.iloc[v_idx]['name']
            j_name = (df.iloc[j_idx]['name'][:30] + "...") if len(df.iloc[j_idx]['name']) > 30 else df.iloc[j_idx]['name']
            v_score = vibe_sims[v_idx]
            j_score = 1.0 - jsd_distances[j_idx]
            
            row = "{:<2}. {:<30} ({:.3f}) | {:<2}. {:<30} ({:.3f})".format(
                i+1, v_name, v_score, i+1, j_name, j_score
            )
            print(row)

if __name__ == "__main__":
    interactive_search()
