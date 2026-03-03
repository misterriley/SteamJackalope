import pickle
import numpy as np
import pandas as pd
import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import TOPIC_MODEL_FILE, TOPIC_DISTRIBUTIONS_FILE, METADATA_FILE, MODEL_NAME

def map_topics_to_motivations():
    motivations = [
        "Destruction", "Excitement", "Competition", "Community", 
        "Challenge", "Strategy", "Completion", "Power", 
        "Fantasy", "Story", "Discovery", "Design"
    ]
    
    print("Loading topic model and embeddings...")
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    
    topic_embeddings = topic_model.topic_embeddings_[1:] 
    
    embed_model = SentenceTransformer(MODEL_NAME)
    motivation_embeddings = embed_model.encode(motivations).astype(np.float32)
    
    print("Calculating Topic-Motivation Similarity...")
    sim_matrix = cosine_similarity(motivation_embeddings, topic_embeddings)
    
    topic_to_motivation_idx = np.argmax(sim_matrix, axis=0)
    
    motivation_to_topics = {i: [] for i in range(len(motivations))}
    for topic_idx, mot_idx in enumerate(topic_to_motivation_idx):
        motivation_to_topics[mot_idx].append(topic_idx)
        
    print("Loading topic distributions and metadata...")
    all_probs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    df = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    topic_info = topic_model.get_topic_info()
    
    results = []
    
    for mot_idx, mot_name in enumerate(motivations):
        assigned_topic_indices = motivation_to_topics[mot_idx]
        
        if not assigned_topic_indices:
            results.append({
                'Motivation': mot_name,
                'Topic Count': 0,
                'Primary Topics': "None",
                'Top Games': "None"
            })
            continue
            
        cat_scores = np.sum(all_probs[:, assigned_topic_indices], axis=1)
        
        topic_sims = sim_matrix[mot_idx, assigned_topic_indices]
        top_topic_indices_in_mot = np.argsort(-topic_sims)[:5]
        
        display_topics = []
        for i in top_topic_indices_in_mot:
            t_idx = assigned_topic_indices[i]
            t_row = topic_info.iloc[t_idx+1]
            label = t_row['Name'].split('_', 1)[1] if '_' in t_row['Name'] else t_row['Name']
            display_topics.append("T" + str(t_row['Topic']) + " (" + label + ")")

        top_game_indices = np.argsort(-cat_scores)[:5]
        display_games = []
        for idx in top_game_indices:
            game = df.iloc[idx]
            display_games.append(str(game['name']) + " (" + str(round(float(cat_scores[idx]), 4)) + ")")
            
        results.append({
            'Motivation': mot_name,
            'Topic Count': len(assigned_topic_indices),
            'Primary Topics': ", ".join(display_topics),
            'Top Games': " | ".join(display_games)
        })

    print("")
    print("="*100)
    print("Motivation   | Topics | Primary Topics (Representative)")
    print("-" * 100)
    for res in results:
        line = res['Motivation'].ljust(12) + " | " + str(res['Topic Count']).ljust(6) + " | " + res['Primary Topics']
        print(line[:100])
    print("="*100)
    
    print("")
    print("Top Games per Motivation:")
    for res in results:
        print("")
        print("[" + res['Motivation'] + "]")
        print("Games: " + res['Top Games'])

if __name__ == "__main__":
    map_topics_to_motivations()
