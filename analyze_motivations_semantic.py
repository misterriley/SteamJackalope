import numpy as np
import pandas as pd
import pickle
import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import json

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TOPIC_DISTRIBUTIONS_FILE, 
    EMBEDDINGS_DESC_FILE, 
    W_DESC_FILE, 
    MEAN_DESC_FILE, 
    METADATA_FILE, 
    MODEL_NAME,
    ROOT_DIR,
    TOPIC_MODEL_FILE
)

def analyze():
    motivations = ["Destruction", "Excitement", "Competition", "Community", "Challenge", "Strategy", "Completion", "Power", "Fantasy", "Story", "Discovery", "Design"]
    W = np.load(W_DESC_FILE)
    mean = np.load(MEAN_DESC_FILE)
    embed_model = SentenceTransformer(MODEL_NAME)
    mot_raw = embed_model.encode(motivations).astype(np.float32)
    mot_whitened = (mot_raw - mean) @ W
    topic_probs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    sem_whitened = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    
    num_topics = topic_probs.shape[1]
    topic_centroids = np.zeros((num_topics, 235), dtype=np.float32)
    p_sums = np.zeros(num_topics, dtype=np.float32)
    batch_size = 20000
    for i in range(0, topic_probs.shape[0], batch_size):
        end = min(i + batch_size, topic_probs.shape[0])
        p_batch = topic_probs[i:end].astype(np.float32)
        s_batch = sem_whitened[i:end].astype(np.float32)
        topic_centroids += p_batch.T @ s_batch
        p_sums += np.sum(p_batch, axis=0)
    topic_centroids = topic_centroids / (p_sums[:, np.newaxis] + 1e-9)
    topic_mot_sim = topic_centroids @ mot_whitened.T 
    topic_to_mot = np.argmax(topic_mot_sim, axis=1)
    df_meta = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    topic_info = topic_model.get_topic_info()
    LABELS_FILE = os.path.join(ROOT_DIR, "data", "production", "semantic_dimension_labels.json")
    sem_labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            sem_labels = json.load(f)

    for m_idx, m_name in enumerate(motivations):
        print("\n" + "="*80)
        print("MOTIVATION: " + m_name)
        m_vec = mot_whitened[m_idx]
        top_pos = np.argsort(-m_vec)[:3]
        top_neg = np.argsort(m_vec)[:3]
        print("Top Semantic Dimensions:")
        for d in top_pos:
            l = sem_labels.get(str(d), {}).get('dynamic_label', "Dim " + str(d))
            print("  (+) " + l + " (" + str(round(float(m_vec[d]), 3)) + ")")
        for d in top_neg:
            l = sem_labels.get(str(d), {}).get('dynamic_label', "Dim " + str(d))
            print("  (-) " + l + " (" + str(round(float(m_vec[d]), 3)) + ")")
        assigned = [i for i, v in enumerate(topic_to_mot) if v == m_idx]
        print("Assigned Topics (" + str(len(assigned)) + "):")
        a_sims = topic_mot_sim[assigned, m_idx]
        top_a_indices = np.argsort(-a_sims)[:5]
        for i in top_a_indices:
            t_idx = assigned[i]
            t_row = topic_info.iloc[t_idx+1]
            t_name = t_row['Name'].split('_', 1)[1] if '_' in t_row['Name'] else t_row['Name']
            print("  T" + str(t_row['Topic']) + ": " + t_name)
        cat_game_scores = np.sum(topic_probs[:, assigned], axis=1)
        top_g = np.argsort(-cat_game_scores)[:5]
        print("Top Representative Games:")
        for idx in top_g:
            game = df_meta.iloc[idx]
            print("  - " + str(game['name']) + " (" + str(round(float(cat_game_scores[idx]), 4)) + ")")

if __name__ == "__main__":
    analyze()
