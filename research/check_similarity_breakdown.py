import pandas as pd
import numpy as np
import os
import sys
import re
import ast

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    METADATA_FILE, PRODUCTION_DATA_DIR, EMBEDDINGS_DESC_FILE, 
    EMBEDDINGS_DESC_NORMS_FILE, TOPIC_DISTRIBUTIONS_FILE,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)
from common.utils import calculate_jackalope_kernel, MIGS, STRUCTURAL_MIGS, SEMI_STRUCTURAL_MIGS

def breakdown_similarity():
    nier_appid = 524220
    stellar_appid = 3489700
    
    df = pd.read_parquet(METADATA_FILE)
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(df['appid'])}
    n_idx = appid_to_idx[nier_appid]
    s_idx = appid_to_idx[stellar_appid]
    
    all_verbs = np.load(os.path.join(PRODUCTION_DATA_DIR, "diffused_verb_profiles.npy"), mmap_mode='r').astype(np.float32)
    all_sem = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r').astype(np.float32)
    all_sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r').astype(np.float32)
    all_topics = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r').astype(np.float32)
    all_graph = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_graph.npy'), mmap_mode='r').astype(np.float32)

    # 1. Manual Component Check
    # A. Semantic Sim
    sem_nier = all_sem[n_idx]
    sem_stellar = all_sem[s_idx]
    norm_n = all_sem_norms[n_idx]
    norm_s = all_sem_norms[s_idx]
    l = SEMANTIC_DOT_PRODUCT_LAMBDA
    sem_sim = (np.dot(sem_nier, sem_stellar) / (norm_n + l)) / (norm_s + l)
    
    # B. Verb Jaccard
    v_n = all_verbs[n_idx]
    v_s = all_verbs[s_idx]
    verb_sim = np.sum(np.minimum(v_n, v_s)) / (np.sum(np.maximum(v_n, v_s)) + 1e-9)
    
    # C. Graph Sim
    g_n = all_graph[n_idx]
    g_s = all_graph[s_idx]
    graph_sim = np.dot(g_n, g_s) / (np.linalg.norm(g_n) * np.linalg.norm(g_s) + 1e-9)
    
    # D. Identity Match
    tags_n_str = df.iloc[n_idx]['tags']
    tags_s_str = df.iloc[s_idx]['tags']
    
    tag_nier = set(ast.literal_eval(tags_n_str).keys()) if isinstance(tags_n_str, str) else set()
    tag_stellar = set(ast.literal_eval(tags_s_str).keys()) if isinstance(tags_s_str, str) else set()
    
    n_migs = [g for g, tags in MIGS.items() if any(t in tag_nier for t in tags)]
    s_migs = [g for g, tags in MIGS.items() if any(t in tag_stellar for t in tags)]
    
    inter = set(n_migs).intersection(set(s_migs))
    union = set(n_migs).union(set(s_migs))
    # Standard Jaccard for identity
    id_match = len(inter) / len(union) if union else 1.0

    print(f"--- SIMILARITY BREAKDOWN: NIER vs STELLAR ---")
    print(f"Semantic Sim: {sem_sim:.4f}")
    print(f"Verb Jaccard: {verb_sim:.4f}")
    print(f"Graph Sim:    {graph_sim:.4f}")
    print(f"Identity Jaccard: {id_match:.4f}")
    print(f"Shared MIGs: {inter}")
    print(f"NieR Only MIGs: {set(n_migs) - inter}")
    print(f"Stellar Only MIGs: {set(s_migs) - inter}")

if __name__ == '__main__':
    breakdown_similarity()
