import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from sentence_transformers import SentenceTransformer
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    MODEL_NAME, W_DESC_FILE, MEAN_DESC_FILE, 
    EMBEDDINGS_DESC_FILE, TOPIC_DISTRIBUTIONS_FILE, 
    METADATA_FILE, TAG_NAMES_FILE, TOPIC_MODEL_FILE,
    ROOT_DIR, PRODUCTION_DATA_DIR, TAG_PRIOR_COUNTS_FILE
)

def run_synthetic_analysis():
    profile_path = "research/synthetic_motivation_tags_fixed.json"
    if not os.path.exists(profile_path):
        print("Error: synthetic profiles not found.")
        return
        
    with open(profile_path, 'r') as f:
        profiles = json.load(f)
        
    reg_path = os.path.join(PRODUCTION_DATA_DIR, "regularization_constants.json")
    with open(reg_path, 'r') as f:
        reg = json.load(f)
        
    K = reg.get("TAG_VECTOR_K", 100.0)
    G_prior = np.load(TAG_PRIOR_COUNTS_FILE) 
    
    with open(TAG_NAMES_FILE, 'r') as f:
        tag_names = json.load(f)
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    
    embed_model = SentenceTransformer(MODEL_NAME)
    W_sem = np.load(W_DESC_FILE).astype(np.float32)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    topic_probs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    topic_embeddings = topic_model.topic_embeddings_[1:]
    
    w_tag_path = os.path.join(PRODUCTION_DATA_DIR, "w_tag.npy")
    W_tag = np.load(w_tag_path).astype(np.float32)
    tag_vectors_prod = np.load(os.path.join(PRODUCTION_DATA_DIR, "steam_tag_vectors.npy"), mmap_mode='r')
    tag_norms_prod = np.load(os.path.join(PRODUCTION_DATA_DIR, "tag_vectors_norms.npy"), mmap_mode='r').reshape(-1, 1)
    
    K_white = tag_vectors_prod.shape[1]
    if W_tag.shape[1] > K_white:
        W_tag = W_tag[:, :K_white]

    results = []

    def standardize(scores):
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-9: return scores - mean
        return (scores - mean) / std

    for mot_name, data in profiles.items():
        print("Analyzing: " + mot_name)
        tags = data['synthetic_tags']
        c_m = np.zeros(len(tag_names))
        for t in tags:
            if t in tag_to_idx: c_m[tag_to_idx[t]] = 1.0
            
        n_m = np.sum(c_m)
        if n_m == 0:
            tag_scores = np.zeros(len(df_meta))
        else:
            p_m = (c_m + K * G_prior) / (n_m + K)
            log_p = np.log(p_m + 1e-9)
            v_m_raw = log_p - np.mean(log_p)
            log_G = np.log(G_prior + 1e-9)
            v_G = log_G - np.mean(log_G)
            v_m_centered = v_m_raw - v_G
            v_white = np.dot(v_m_centered, W_tag)
            tag_scores = np.dot(tag_vectors_prod.astype(np.float32), v_white).flatten()
            tag_scores /= (tag_norms_prod.flatten() + 1e-9)

        full_text = mot_name + ". " + data['description']
        sem_raw = embed_model.encode([full_text]).astype(np.float32)
        sem_whitened = sem_raw @ W_sem
        sem_norm = np.linalg.norm(sem_whitened)
        if sem_norm > 1e-9: sem_whitened /= sem_norm
        sem_scores = np.dot(sem_vectors.astype(np.float32), sem_whitened.T).flatten()
        
        sims_to_topics = cosine_similarity(sem_raw, topic_embeddings)[0]
        T_temp = 0.2
        sims_to_topics = sims_to_topics - np.max(sims_to_topics)
        exp_sim = np.exp(sims_to_topics / T_temp)
        mot_topic_probs = exp_sim / np.sum(exp_sim)
        topic_scores = np.dot(topic_probs.astype(np.float32), mot_topic_probs)
        
        # Standardize before blending
        z_tag = standardize(tag_scores)
        z_sem = standardize(sem_scores)
        z_top = standardize(topic_scores)
        
        # Equal weight blend (1/3 each)
        hybrid_scores = (z_tag + z_sem + z_top) / 3.0
        
        top_game_indices = np.argsort(-hybrid_scores)[:10]
        top_games = [df_meta.iloc[idx]['name'] for idx in top_game_indices]
        
        results.append({'motivation': mot_name, 'synthetic_tags': tags, 'top_games': top_games})

    print("\n" + "="*80)
    print("HYBRID SYNTHETIC MOTIVATION ANALYSIS (STANDARDIZED BLEND)")
    print("="*80)
    
    for res in results:
        print("\n[" + res['motivation'].upper() + "]")
        print("Input Tags: " + ", ".join(res['synthetic_tags']))
        print("Top Games: " + " | ".join(res['top_games'][:5]))

if __name__ == "__main__":
    run_synthetic_analysis()
