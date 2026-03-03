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
    ROOT_DIR, PRODUCTION_DATA_DIR
)
import pipeline.generate_tag_vectors as gtv

def analyze_motivations():
    csv_path = "research/GamerMotivationDescriptions.csv"
    if not os.path.exists(csv_path):
        print("Error: " + csv_path + " not found.")
        return
    
    print("Loading motivation descriptions...")
    # Standard CSV now
    df_mot = pd.read_csv(csv_path, encoding='latin1')
    print("Loaded " + str(len(df_mot)) + " motivations.")

    print("Loading models and data...")
    embed_model = SentenceTransformer(MODEL_NAME)
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    
    W_sem = np.load(W_DESC_FILE).astype(np.float32)
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    topic_probs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    
    reg_path = os.path.join(PRODUCTION_DATA_DIR, "regularization_constants.json")
    with open(reg_path, 'r') as f:
        reg = json.load(f)
    
    sem_scale = reg.get("SEMANTIC_GLOBAL_SCALING_FACTOR", 2.0)
    top_scale = reg.get("TOPIC_GLOBAL_SCALING_FACTOR", 26.96)
    
    print("Generating CLR Tag Vectors for correlation analysis...")
    original_use_whitening = gtv.USE_TAG_WHITENING
    gtv.USE_TAG_WHITENING = False
    temp_v = "research/temp_clr_for_mot.npy"
    temp_c = "research/temp_clr_const_for_mot.json"
    clr_vectors, _ = gtv.generate_tag_vectors("data/pipeline_games_clean.csv", output_vectors=temp_v, output_constants=temp_c)
    gtv.USE_TAG_WHITENING = original_use_whitening
    
    print("Standardizing tags...")
    T = clr_vectors.astype(np.float32)
    T_mean = np.mean(T, axis=0)
    T_std = np.std(T, axis=0)
    T_std[T_std < 1e-9] = 1.0
    T_z = (T - T_mean) / T_std
    
    with open(TAG_NAMES_FILE, 'r') as f:
        tag_names = json.load(f)

    results = []

    for _, row in df_mot.iterrows():
        mot_name = str(row['Motivation'])
        short_desc = str(row['Short Description'])
        long_desc = str(row['Long Description']).replace('\n', ' ').replace('\r', '')
        # Clean up common encoding artifacts
        long_desc = long_desc.replace('', "'")
        
        full_text = mot_name + ". " + short_desc + ". " + long_desc
        
        print("\nAnalyzing Motivation: " + mot_name)
        
        # 1. Semantic Embedding
        sem_raw = embed_model.encode([full_text]).astype(np.float32)
        sem_whitened = sem_raw @ W_sem
        sem_norm = np.linalg.norm(sem_whitened)
        if sem_norm > 1e-9:
            sem_whitened /= sem_norm
            
        # 2. Topic Transformation
        topic_embeddings = topic_model.topic_embeddings_[1:] 
        sims_to_topics = cosine_similarity(sem_raw, topic_embeddings)[0]
        
        T_temp = 0.2
        sims_to_topics = sims_to_topics - np.max(sims_to_topics)
        exp_sim = np.exp(sims_to_topics / T_temp)
        mot_topic_probs = exp_sim / np.sum(exp_sim)
        
        # 3. Score Games
        s_scores = np.dot(sem_vectors.astype(np.float32), sem_whitened.T).flatten()
        t_scores = np.dot(topic_probs.astype(np.float32), mot_topic_probs)
        hybrid_scores = (s_scores * sem_scale) + (t_scores * top_scale)
        
        # 4. Tag Correlation
        h_mean = np.mean(hybrid_scores)
        h_std = np.std(hybrid_scores)
        if h_std < 1e-9: h_std = 1.0
        h_z = (hybrid_scores - h_mean) / h_std
        
        tag_corrs = np.dot(T_z.T, h_z) / len(h_z)
        top_tag_indices = np.argsort(-tag_corrs)[:15]
        top_tags = [(tag_names[i], float(tag_corrs[i])) for i in top_tag_indices]
        
        # 5. Top Games
        top_game_indices = np.argsort(-hybrid_scores)[:10]
        top_games = []
        for idx in top_game_indices:
            game = df_meta.iloc[idx]
            top_games.append({
                'name': game['name'],
                'appid': int(game['appid']),
                'score': float(hybrid_scores[idx])
            })
            
        results.append({
            'motivation': mot_name,
            'top_tags': top_tags,
            'top_games': top_games,
            'avg_score': float(np.mean(hybrid_scores)),
            'std_score': float(np.std(hybrid_scores))
        })

    print("\n" + "="*80)
    print("GAMER MOTIVATION ANALYSIS REPORT")
    print("="*80)
    
    for res in results:
        print("\n[" + res['motivation'].upper() + "]")
        print("Stats: Avg=" + str(round(res['avg_score'], 4)) + ", Std=" + str(round(res['std_score'], 4)))
        tag_str = ", ".join([t + "(" + str(round(s, 2)) + ")" for t, s in res['top_tags'] if s > 0.1])
        print("Top Tags (>0.1): " + tag_str)
        print("Top Games:")
        for i, g in enumerate(res['top_games'][:5]):
            print("  " + str(i+1) + ". " + g['name'] + " (Score: " + str(round(g['score'], 4)) + ")")

    if os.path.exists(temp_v): os.remove(temp_v)
    if os.path.exists(temp_c): os.remove(temp_c)

if __name__ == "__main__":
    analyze_motivations()
