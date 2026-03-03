import pickle
import numpy as np
import pandas as pd
import os
import sys
import warnings

# Suppress warnings from loading old pickles
warnings.filterwarnings("ignore")

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import TOPIC_MODEL_FILE, TOPIC_DISTRIBUTIONS_FILE, METADATA_FILE

def match_prompt_to_topics():
    prompt = "Gamers who score high on this component are agents of chaos and destruction. They love having many tools at their disposal to blow things up and cause relentless mayhem. They enjoy games with lots of guns and explosives."
    
    print(f"Loading topic model from {TOPIC_MODEL_FILE}...")
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    
    print("Transforming prompt into topic space...")
    # Get topic distribution for the prompt
    _, probs_prompt = topic_model.transform([prompt])
    probs_prompt = probs_prompt[0] 
    
    # Get top 5 activated topics
    top_topic_indices = np.argsort(-probs_prompt)[:5]
    
    print("\nTop Activated Topics for the Prompt:")
    for idx in top_topic_indices:
        topic_info = topic_model.get_topic_info().iloc[idx+1]
        topic_id = topic_info['Topic']
        label = topic_info['Name']
        prob = probs_prompt[idx]
        print(f"Topic {topic_id}: {label} (Probability: {prob:.4f})")

    print("\nCalculating game scores based on this topic distribution...")
    all_probs = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    scores = np.dot(all_probs.astype(np.float32), probs_prompt.astype(np.float32))
    
    print("Loading metadata...")
    df = pd.read_parquet(METADATA_FILE, columns=['appid', 'name'])
    
    top_10_indices = np.argsort(-scores)[:10]
    
    print("\nTop 10 Games matching the Destruction/Chaos prompt (Topic Space):")
    for i, idx in enumerate(top_10_indices):
        game = df.iloc[idx]
        print(f"{i+1}. {game['name']} (AppID: {game['appid']}, Score: {scores[idx]:.4f})")

if __name__ == "__main__":
    match_prompt_to_topics()
