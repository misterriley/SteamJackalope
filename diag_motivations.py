import pickle
import numpy as np
import pandas as pd
import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import TOPIC_MODEL_FILE, MODEL_NAME

def diag():
    motivations = ["Destruction", "Excitement", "Competition", "Community", "Challenge", "Strategy", "Completion", "Power", "Fantasy", "Story", "Discovery", "Design"]
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    topic_embeddings = topic_model.topic_embeddings_[1:] 
    embed_model = SentenceTransformer(MODEL_NAME)
    motivation_embeddings = embed_model.encode(motivations).astype(np.float32)
    sim_matrix = cosine_similarity(motivation_embeddings, topic_embeddings)
    
    print("Max Similarity per Motivation:")
    for i, mot in enumerate(motivations):
        max_s = float(np.max(sim_matrix[i]))
        print(mot + ": " + str(round(max_s, 4)))

    print("\nAssignment Counts:")
    topic_to_mot = np.argmax(sim_matrix, axis=0)
    counts = np.bincount(topic_to_mot, minlength=len(motivations))
    for i, mot in enumerate(motivations):
        print(mot + ": " + str(counts[i]))

if __name__ == "__main__":
    diag()
