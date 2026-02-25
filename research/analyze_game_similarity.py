import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, EMBEDDINGS_DESC_FILE, TOPIC_DISTRIBUTIONS_FILE,
    TAG_NORMS_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR, SEMANTIC_GLOBAL_SCALING_FACTOR, TOPIC_GLOBAL_SCALING_FACTOR
)

def analyze_similarity(id1=2416100, id2=57300):
    df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name'])
    appid_to_idx = {aid: i for i, aid in enumerate(df['appid'])}
    
    idx1, idx2 = appid_to_idx[id1], appid_to_idx[id2]
    
    # 1. Tags
    tags = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    
    t1 = tags[idx1] / (norms[idx1] + DOT_PRODUCT_LAMBDA)
    t2 = tags[idx2] / (norms[idx2] + DOT_PRODUCT_LAMBDA)
    tag_sim = np.dot(t1, t2) * TAG_GLOBAL_SCALING_FACTOR
    
    # 2. Semantics
    sems = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    s_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    
    s1 = sems[idx1] / (s_norms[idx1] + SEMANTIC_DOT_PRODUCT_LAMBDA)
    s2 = sems[idx2] / (s_norms[idx2] + SEMANTIC_DOT_PRODUCT_LAMBDA)
    sem_sim = np.dot(s1, s2) * SEMANTIC_GLOBAL_SCALING_FACTOR
    
    # 3. Topics
    topics = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    top1 = topics[idx1]
    top2 = topics[idx2]
    topic_sim = np.dot(top1, top2) * TOPIC_GLOBAL_SCALING_FACTOR
    
    print("\n--- Similarity Breakdown ---")
    print("Tag Similarity:      " + f"{tag_sim:.4f}")
    print("Semantic Similarity: " + f"{sem_sim:.4f}")
    print("Topic Similarity:    " + f"{topic_sim:.4f}")
    print("Total Thematic:      " + f"{tag_sim + sem_sim + topic_sim:.4f}")

    # Find Top Shared Semantic Dimensions
    s_impact = s1 * s2 * SEMANTIC_GLOBAL_SCALING_FACTOR
    top_s_dims = np.argsort(-np.abs(s_impact))[:5]
    print("\nTop Shared Semantic Dims (Impact):")
    for d in top_s_dims:
        print("  Dim " + str(d) + ": " + f"{s_impact[d]:.4f}")

    # Find Top Shared Topics
    top_shared_topics = np.argsort(-(top1 * top2))[:5]
    print("\nTop Shared Topics (Overlap):")
    for t in top_shared_topics:
        overlap = top1[t] * top2[t] * TOPIC_GLOBAL_SCALING_FACTOR
        if overlap > 0.0001:
            print("  Topic " + str(t) + ": " + f"{overlap:.4f}" + " (Prob1: " + f"{top1[t]:.3f}" + ", Prob2: " + f"{top2[t]:.3f}" + ")")

if __name__ == "__main__":
    analyze_similarity()
