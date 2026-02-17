import pandas as pd
import numpy as np
import scipy.sparse as sp
import ast
from tqdm import tqdm
import os

def check_rank():
    df = pd.read_csv('data/pipeline_games_clean.csv', low_memory=False)
    print(f"Games: {len(df)}")
    
    # Just take a subset to speed up
    subset = df[df['positive'] + df['negative'] > 100].head(10000)
    print(f"Subset for rank check: {len(subset)}")
    
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
    print(f"Tags in subset: {num_tags}")
    
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
    
    # Normalize
    row_sums = dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    profiles = dense / row_sums
    
    # Check rank
    rank = np.linalg.matrix_rank(profiles)
    print(f"Rank of profiles: {rank}")
    
    # SVD
    U, S, Vt = np.linalg.svd(profiles, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1
    
    print(f"Components for 95% variance: {n_95}")
    print(f"Components for 99% variance: {n_99}")

if __name__ == "__main__":
    check_rank()
