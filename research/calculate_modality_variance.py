import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import (
    TAG_VECTORS_FILE, EMBEDDINGS_DESC_FILE, TOPIC_DISTRIBUTIONS_FILE,
    TAG_NORMS_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    DOT_PRODUCT_LAMBDA, SEMANTIC_DOT_PRODUCT_LAMBDA, REGULARIZATION_FILE
)

def calculate_population_variances():
    print("Loading data for population variance study...")
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r').reshape(-1, 1)
    
    sample_size = min(len(tag_vectors), 10000)
    indices = np.random.choice(len(tag_vectors), sample_size, replace=False)
    
    tag_sample = tag_vectors[indices]
    tag_norm_sample = tag_norms[indices]
    tag_feat = tag_sample / (tag_norm_sample + DOT_PRODUCT_LAMBDA)
    tag_var = np.mean(np.var(tag_feat, axis=0))
    
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').reshape(-1, 1)
    sem_sample = sem_vectors[indices]
    sem_norm_sample = sem_norms[indices]
    sem_feat = sem_sample / (sem_norm_sample + SEMANTIC_DOT_PRODUCT_LAMBDA)
    sem_var = np.mean(np.var(sem_feat, axis=0))
    
    topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')[indices].astype(np.float32)
    topic_var = np.mean(np.var(topic_dist, axis=0))
    
    print("\n--- Population-Wide Variance per Dimension (Unscaled) ---")
    print("Tags:      " + f"{tag_var:.8f}")
    print("Semantics: " + f"{sem_var:.8f}")
    print("Topics:    " + f"{topic_var:.8f}")
    
    sem_mult = float(np.sqrt(tag_var / sem_var))
    top_mult = float(np.sqrt(tag_var / topic_var))
    
    print("\n--- Multipliers Needed to Match Tag Variance ---")
    print("Semantic Multiplier: " + f"{sem_mult:.4f}" + "x")
    print("Topic Multiplier:    " + f"{top_mult:.4f}" + "x")

    # Update regularization_constants.json
    import json
    reg_data = {}
    if os.path.exists(REGULARIZATION_FILE):
        with open(REGULARIZATION_FILE, "r") as f:
            reg_data = json.load(f)
            
    reg_data["TAG_GLOBAL_SCALING_FACTOR"] = 1.0
    reg_data["SEMANTIC_GLOBAL_SCALING_FACTOR"] = round(sem_mult, 2)
    reg_data["TOPIC_GLOBAL_SCALING_FACTOR"] = round(top_mult, 2)
    
    with open(REGULARIZATION_FILE, "w") as f:
        json.dump(reg_data, f, indent=4)
    print("\nSaved multipliers to " + REGULARIZATION_FILE)

if __name__ == "__main__":
    calculate_population_variances()
