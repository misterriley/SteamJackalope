import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

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

def create_motivations_library():
    profile_path = "research/synthetic_motivation_tags_fixed.json"
    if not os.path.exists(profile_path):
        print("Error: synthetic profiles not found.")
        return
        
    with open(profile_path, 'r') as f:
        profiles = json.load(f)
        
    print("Loading models and artifacts...")
    embed_model = SentenceTransformer(MODEL_NAME)
    W_sem = np.load(W_DESC_FILE).astype(np.float32)
    
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    topic_embeddings = topic_model.topic_embeddings_[1:] 
    
    with open(TAG_NAMES_FILE, 'r') as f:
        tag_names = json.load(f)
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    
    w_tag_path = os.path.join(PRODUCTION_DATA_DIR, "w_tag.npy")
    W_tag = np.load(w_tag_path).astype(np.float32)
    G_prior = np.load(TAG_PRIOR_COUNTS_FILE)
    
    reg_path = os.path.join(PRODUCTION_DATA_DIR, "regularization_constants.json")
    with open(reg_path, 'r') as f:
        reg = json.load(f)
    K = reg.get("TAG_VECTOR_K", 100.0)
    
    library = {}

    print("Generating motivation vectors...")
    for mot_name, data in profiles.items():
        # A. Semantic
        text = mot_name + ". " + data['description']
        s_raw = embed_model.encode([text]).astype(np.float32)
        s_white = s_raw @ W_sem
        s_norm = np.linalg.norm(s_white)
        if s_norm > 1e-9: s_white /= s_norm
        
        # B. Topic
        sims = cosine_similarity(s_raw, topic_embeddings)[0]
        T_temp = 0.2
        sims = sims - np.max(sims)
        exp_sim = np.exp(sims / T_temp)
        t_dist = exp_sim / np.sum(exp_sim)
        
        # C. Tag
        tags = data['synthetic_tags']
        c_m = np.zeros(len(tag_names))
        for t in tags:
            if t in tag_to_idx: c_m[tag_to_idx[t]] = 1.0
        n_m = np.sum(c_m)
        if n_m > 0:
            p_m = (c_m + K * G_prior) / (n_m + K)
            log_p = np.log(p_m + 1e-9)
            v_m_raw = log_p - np.mean(log_p)
            log_G = np.log(G_prior + 1e-9)
            v_G = log_G - np.mean(log_G)
            v_m_centered = v_m_raw - v_G
            v_white = np.dot(v_m_centered, W_tag)
            # Unit normalize in full white space
            v_norm = np.linalg.norm(v_white)
            if v_norm > 1e-9: v_white /= v_norm
        else:
            v_white = np.zeros(W_tag.shape[1])

        library[mot_name] = {
            "semantic_vector": s_white.flatten().tolist(),
            "topic_vector": t_dist.flatten().tolist(),
            "tag_vector": v_white.flatten().tolist(),
            "tags": tags
        }

    output_file = os.path.join(PRODUCTION_DATA_DIR, "motivations_library.json")
    with open(output_file, 'w') as f:
        json.dump(library, f, indent=4)
    print("Saved library to " + output_file)
    return library

def analyze_user_dna(steamid, library):
    profile_path = "data/user_" + steamid + "_taste_profile.json"
    if not os.path.exists(profile_path):
        print("Error: profile not found.")
        return
        
    print("Loading Taste DNA...")
    with open(profile_path, 'r') as f:
        user = json.load(f)
        
    u_tag = np.array(user['vibe_vector'])
    u_sem = np.array(user['semantic_vibe_vector'])
    u_top = np.array(user['topic_vibe_vector'])
    w_tag = user['beta']
    w_sem = user['alpha']
    w_top = user['gamma_topic']
    
    # Load topic standardization stats to project motivation topics into taste space
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy"))
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy"))
    
    motivation_scores = []
    for mot_name, vectors in library.items():
        # Tag: Truncate motivation tag vector to match user vibe space
        m_tag_full = np.array(vectors['tag_vector'])
        m_tag = m_tag_full[:len(u_tag)]
        # Re-normalize after truncation to ensure fair dot product
        m_tag_norm = np.linalg.norm(m_tag)
        if m_tag_norm > 1e-9: m_tag /= m_tag_norm
        
        m_sem = np.array(vectors['semantic_vector'])
        
        # Topic: motivation topic is a raw probability dist. 
        # User profile topic_vibe_vector is a unit vector in STANDARDIZED topic space.
        m_top_raw = np.array(vectors['topic_vector'])
        # Project motivation topic dist into standardized space
        m_top_std = (m_top_raw - t_means) / (t_stds + 1e-9)
        # Unit normalize in standardized space to match user vibe
        m_top_norm = np.linalg.norm(m_top_std)
        if m_top_norm > 1e-9: m_top_std /= m_top_norm
        
        s_tag = np.dot(u_tag, m_tag)
        s_sem = np.dot(u_sem, m_sem)
        s_top = np.dot(u_top, m_top_std)
        
        total_score = (s_tag * w_tag) + (s_sem * w_sem) + (s_top * w_top)
        
        motivation_scores.append({
            "motivation": mot_name,
            "score": float(total_score),
            "tag_sim": float(s_tag),
            "sem_sim": float(s_sem),
            "top_sim": float(s_top)
        })
        
    motivation_scores = sorted(motivation_scores, key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*80)
    print("USER MOTIVATION PROFILE: " + steamid)
    print("="*80)
    for m in motivation_scores:
        line = m['motivation'].ljust(12) + ": " + str(round(m['score'], 2)).rjust(6)
        line += " (T:" + str(round(m['tag_sim'], 2)).rjust(5)
        line += ", S:" + str(round(m['sem_sim'], 2)).rjust(5)
        line += ", P:" + str(round(m['top_sim'], 2)).rjust(5) + ")"
        print(line)
    print("="*80)

if __name__ == "__main__":
    lib = create_motivations_library()
    if len(sys.argv) > 1:
        analyze_user_dna(sys.argv[1], lib)
