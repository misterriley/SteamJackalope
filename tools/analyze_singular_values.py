import pandas as pd
import numpy as np
import scipy.sparse as sp
import ast
import os

def analyze_s_distribution():
    print("Loading data for SVD analysis...")
    df = pd.read_csv('data/pipeline_games_clean.csv', low_memory=False)
    subset = df[df['positive'] + df['negative'] > 100]
    
    all_game_tags = []
    global_tags = set()
    for tag_str in subset['tags']:
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '': continue
        try:
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                all_game_tags.append(tags_dict)
                global_tags.update(tags_dict.keys())
        except: continue
            
    unique_tags = sorted(list(global_tags))
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    num_tags = len(unique_tags)
    
    data = []
    row_ind = []
    col_ind = []
    for i, tags in enumerate(all_game_tags):
        for t, c in tags.items():
            row_ind.append(i)
            col_ind.append(tag_to_idx[t])
            data.append(c)
            
    sparse = sp.csr_matrix((data, (row_ind, col_ind)), shape=(len(all_game_tags), num_tags))
    dense = sparse.toarray()
    
    # Bayesian Regularization (simplified)
    K = 68.0
    G = dense.sum(axis=0)
    G = G / G.sum()
    N = dense.sum(axis=1, keepdims=True)
    reg_profiles = (dense + K * G) / (N + K)
    
    # CLR
    log_v = np.log(reg_profiles + 1e-9)
    gm_log = log_v.mean(axis=1, keepdims=True)
    vectors = log_v - gm_log
    
    # Centering
    vectors = vectors - vectors.mean(axis=0)
    
    # SVD
    M = np.dot(vectors.T, vectors) / len(vectors)
    U, S, Vt = np.linalg.svd(M)
    
    print("\n--- Singular Value Distribution ---")
    print(f"Total Components: {len(S)}")
    print(f"Top 5: {S[:5]}")
    print(f"Bottom 5: {S[-5:]}")
    
    cumvar = np.cumsum(S) / np.sum(S)
    
    p80 = np.argmax(cumvar >= 0.80) + 1
    p90 = np.argmax(cumvar >= 0.90) + 1
    p95 = np.argmax(cumvar >= 0.95) + 1
    p99 = np.argmax(cumvar >= 0.99) + 1
    
    print(f"80% Variance: {p80} components (S_min: {S[p80-1]:.4f})")
    print(f"90% Variance: {p90} components (S_min: {S[p90-1]:.4f})")
    print(f"95% Variance: {p95} components (S_min: {S[p95-1]:.4f})")
    print(f"99% Variance: {p99} components (S_min: {S[p99-1]:.4f})")
    
    cond = S[0] / S[-2]
    print(f"Condition Number (excluding last): {cond:.4f}")
    
    print("\nDecile Sample (S values):")
    indices = np.linspace(0, len(S)-2, 11, dtype=int)
    for idx in indices:
        s_val = S[idx]
        boost = 1.0 / np.sqrt(s_val + 1e-6)
        print(f"  Component {idx:3}: S = {s_val:.6f} (Inv-Sqrt boost: {boost:.2f})")

if __name__ == "__main__":
    analyze_s_distribution()
