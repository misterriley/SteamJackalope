import pandas as pd
import numpy as np
import os
import ast
import sys
from tqdm import tqdm

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, TAG_VECTOR_K, GLOBAL_POSITIVE_RATE

def calculate_clr_stats(csv_path="data/pipeline_games_clean.csv"):
    """
    Calculates the mean norm of games in the raw CLR-transformed space.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['tags'])
    
    # Get Global Prior
    print("Estimating global prior...")
    all_counts = []
    for tag_str in df['tags'].dropna().head(10000):
        try:
            d = ast.literal_eval(tag_str)
            if isinstance(d, dict): all_counts.append(d)
        except: continue
    
    tags_set = set()
    for d in all_counts: tags_set.update(d.keys())
    unique_tags = sorted(list(tags_set))
    tag_to_idx = {t: i for i, t in enumerate(unique_tags)}
    
    matrix = np.zeros((len(all_counts), len(unique_tags)))
    for i, d in enumerate(all_counts):
        for t, c in d.items():
            matrix[i, tag_to_idx[t]] = c
    
    prior_G = matrix.sum(axis=0) / matrix.sum()
    prior_G = np.maximum(prior_G, 1e-9)
    log_prior = np.log(prior_G)
    v_prior = log_prior - log_prior.mean()
    
    # Calculate CLR Norms for a sample
    print("Calculating sample CLR norms...")
    norms = []
    K = TAG_VECTOR_K
    
    sample_df = df.dropna(subset=['tags']).sample(min(5000, len(df)))
    for tag_str in tqdm(sample_df['tags']):
        try:
            tags = ast.literal_eval(tag_str)
            if not isinstance(tags, dict): continue
            
            c = np.zeros(len(unique_tags))
            for t, count in tags.items():
                if t in tag_to_idx: c[tag_to_idx[t]] = count
            
            n = c.sum()
            reg_profile = (c + K * prior_G) / (n + K)
            
            log_v = np.log(reg_profile + 1e-9)
            v = log_v - log_v.mean()
            
            final_v = v - v_prior
            norms.append(np.linalg.norm(final_v))
        except: continue
        
    mean_len = np.mean(norms)
    print("\nResults:")
    print(f"Mean CLR Vector Norm: {mean_len:.4f}")
    
    return mean_len

if __name__ == "__main__":
    calculate_clr_stats()
