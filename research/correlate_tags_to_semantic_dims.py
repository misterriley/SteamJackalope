import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv
from common.constants import (
    METADATA_FILE,
    EMBEDDINGS_DESC_FILE,
    TAG_NAMES_FILE
)

def correlate_tags_to_semantic_dims():
    csv_path = "data/pipeline_games_clean.csv"
    if not os.path.exists(csv_path):
        print("Error: " + csv_path + " not found.")
        return

    # 1. Generate Unwhitened CLR Tag Vectors
    print("Generating Unwhitened CLR Tag Vectors...")
    original_use_whitening = gtv.USE_TAG_WHITENING
    gtv.USE_TAG_WHITENING = False
    
    temp_vectors_file = "research/temp_clr_vectors.npy"
    temp_constants_file = "research/temp_clr_constants.json"
    
    clr_vectors, appids = gtv.generate_tag_vectors(csv_path, output_vectors=temp_vectors_file, output_constants=temp_constants_file)
    gtv.USE_TAG_WHITENING = original_use_whitening 
    
    # 2. Load Whitened Semantic Vectors
    print("Loading Whitened Semantic Vectors...")
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    
    if len(clr_vectors) != len(sem_vectors):
        print("Error: Row count mismatch! Tags: " + str(len(clr_vectors)) + ", Semantics: " + str(len(sem_vectors)))
        return

    # 3. Load Tag Names
    with open(TAG_NAMES_FILE, 'r') as f:
        tag_names = json.load(f)
        
    # 4. Calculate Correlation Matrix
    print("Calculating Correlation Matrix (Tags x Semantic Dimensions)...")
    
    def standardize(X):
        X = X.astype(np.float32)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-9] = 1.0
        return (X - mean) / std

    T = standardize(clr_vectors)
    S = standardize(sem_vectors)
    
    R = np.dot(T.T, S) / len(T)
    
    # 5. Extract top correlations per dimension
    num_dims = S.shape[1]
    results = {}
    
    for d in range(num_dims):
        corrs = R[:, d]
        pos_indices = np.argsort(-corrs)[:10]
        top_pos = [(tag_names[idx], float(corrs[idx])) for idx in pos_indices if corrs[idx] > 0.05]
        neg_indices = np.argsort(corrs)[:10]
        top_neg = [(tag_names[idx], float(corrs[idx])) for idx in neg_indices if corrs[idx] < -0.05]
        results[str(d)] = {
            "top_positive": top_pos,
            "top_negative": top_neg
        }
        
    # 6. Save and Print
    output_path = "research/tag_semantic_correlations.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done! Saved correlations to " + output_path)
    
    print("\nSample Tag-Semantic Correlations:")
    for d in range(5):
        print("\nDimension " + str(d) + ":")
        pos = ", ".join([t + "(" + str(round(s, 2)) + ")" for t, s in results[str(d)]['top_positive'][:5]])
        neg = ", ".join([t + "(" + str(round(s, 2)) + ")" for t, s in results[str(d)]['top_negative'][:5]])
        print("  Pos: " + pos)
        print("  Neg: " + neg)

    if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
    if os.path.exists(temp_constants_file): os.remove(temp_constants_file)

if __name__ == "__main__":
    correlate_tags_to_semantic_dims()
