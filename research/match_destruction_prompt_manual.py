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

def match_prompt_to_topics_manual():
    prompt = "Gamers who score high on this component are agents of chaos and destruction. They love having many tools at their disposal to blow things up and cause relentless mayhem. They enjoy games with lots of guns and explosives."
    
    print("Loading topic model...")
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    
    # Valid topic embeddings (skipping -1 outlier)
    topic_embeddings = topic_model.topic_embeddings_[1:]
    
    print("Loading embedding model...")
    embed_model = SentenceTransformer(MODEL_NAME)
    
    print("Embedding prompt...")
    prompt_embedding = embed_model.encode([prompt]).astype(np.float32).reshape(1, -1)
    
    print("Calculating topic probabilities for the prompt...")
    # Calculate Cosine Similarity to centroids
    sims = cosine_similarity(prompt_embedding, topic_embeddings)
    
    # Vector-Space Soft Assignment (Temperature T=0.2)
    T = 0.2
    sims_scaled = (sims - np.max(sims)) / T
    exp_sim = np.exp(sims_scaled)
    probs_prompt = (exp_sim / np.sum(exp_sim))[0]
    
    # Get top 5 activated topics
    top_topic_indices = np.argsort(-probs_prompt)[:5]
    
    print("")
    print("Top Activated Topics for the Prompt:")
    topic_info = topic_model.get_topic_info()
    for idx in top_topic_indices:
        # Offset by 1 in get_topic_info because Topic -1 is first
        t_row = topic_info.iloc[idx+1]
        print("Topic " + str(t_row['Topic']) + ": " + str(t_row['Name']) + " (Prob: " + str(round(float(probs_prompt[idx]), 4)) + ")")

    print("")
    print("Calculating game scores based on this topic distribution...")
    all_probs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    scores = np.dot(all_probs.astype(np.float32), probs_prompt.astype(np.float32))
    
    print("Loading metadata...")
    df = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    
    top_10_indices = np.argsort(-scores)[:10]
    
    print("")
    print("Top 10 Games matching the Destruction/Chaos prompt (Topic Space):")
    for i, idx in enumerate(top_10_indices):
        game = df.iloc[idx]
        print(str(i+1) + ". " + str(game['name']) + " (AppID: " + str(game['appid']) + ", Score: " + str(round(float(scores[idx]), 4)) + ")")

if __name__ == "__main__":
    match_prompt_to_topics_manual()
