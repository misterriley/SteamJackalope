import pandas as pd
import numpy as np
import pickle
import os
import sys
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

def search_jsd_prompt(prompt_text, top_k=10, T=0.05):
    print("\n--- Prompt Search: '" + prompt_text + "' (T=" + str(T) + ") ---")
    
    # 1. Load Model and Data
    df = pd.read_parquet(METADATA_FILE)
    dist_path = os.path.join(ROOT_DIR, "data", "production", "topic_distributions.npy")
    model_path = os.path.join(ROOT_DIR, "data", "production", "topic_model.pkl")
    
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
        
    probs_matrix = np.load(dist_path).astype(np.float32)
    
    # 2. Encode Prompt
    print("Encoding prompt...")
    model = SentenceTransformer(MODEL_NAME)
    prompt_vec = model.encode([prompt_text.lower()])[0].astype(np.float32)
    
    # 3. Map to Topic Space (Vector-Space Soft Assignment)
    topic_embeddings = topic_model.topic_embeddings_[1:] 
    sims = cosine_similarity(prompt_vec.reshape(1, -1), topic_embeddings)[0]
    
    # Softmax
    sims = sims - np.max(sims)
    exp_sim = np.exp(sims / T)
    prompt_p = exp_sim / np.sum(exp_sim)
    
    # 4. Analyze Prompt's Understanding
    print("\n--- Prompt Topic Loadings ---")
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info.Topic != -1]
    
    top_t_indices = np.argsort(-prompt_p)[:5]
    for i in top_t_indices:
        p = prompt_p[i]
        if p > 0.01:
            tid = valid_topics.iloc[i]['Topic']
            words = topic_model.get_topic(tid)
            word_str = ", ".join([w[0] for w in words[:5]])
            print("Topic " + str(tid) + " (" + f"{p:>6.1%}" + "): " + word_str)
            
    # 5. JSD Search
    print("\nCalculating matches...")
    distances = fast_jsd_single(prompt_p, probs_matrix)
    top_indices = np.argsort(distances)[:top_k]
    
    print("\n--- Top " + str(top_k) + " Matches ---")
    print(f"{'Rank':<4} | {'Similarity':<10} | {'Name'}")
    print("-" * 60)
    for i, idx in enumerate(top_indices):
        sim = 1.0 - distances[idx]
        name = df.iloc[idx]['name']
        print(f"{i+1:<4} | {sim:<10.4f} | {name}")

if __name__ == "__main__":
    prompt = "solo horror game recs - preferably indie games"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    search_jsd_prompt(prompt)
