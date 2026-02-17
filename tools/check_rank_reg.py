import pandas as pd
import numpy as np
import scipy.sparse as sp
import ast
import os

def check_rank_with_reg():
    df = pd.read_csv('data/pipeline_games_clean.csv', low_memory=False)
    subset = df[df['positive'] + df['negative'] > 10].head(20000)
    
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
    
    # Bayesian Regularization
    K = 68.0
    G = dense.sum(axis=0)
    G = G / G.sum()
    N = dense.sum(axis=1, keepdims=True)
    
    reg_profiles = (dense + K * G) / (N + K)
    
    # CLR Transform
    log_v = np.log(reg_profiles + 1e-9)
    gm_log = log_v.mean(axis=1, keepdims=True)
    vectors = log_v - gm_log
    
    # Center
    vectors = vectors - vectors.mean(axis=0)
    
    # Check variance
    M = np.dot(vectors.T, vectors) / len(vectors)
    U, S, Vt = np.linalg.svd(M)
    
    cumvar = np.cumsum(S) / np.sum(S)
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1
    
    print(f"Components for 95% variance with CLR + Reg: {n_95}")
    print(f"Components for 99% variance with CLR + Reg: {n_99}")
    print(f"Singular values 1-10: {S[:10]}")
    print(f"Singular values 40-50: {S[40:50]}")

if __name__ == "__main__":
    check_rank_with_reg()
