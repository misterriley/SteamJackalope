import pandas as pd
import numpy as np
import os
import sys
import ast
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, METADATA_FILE, PRODUCTION_DATA_DIR, TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA, TAG_GLOBAL_SCALING_FACTOR, EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE, SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, TOPIC_DISTRIBUTIONS_FILE
)
from common.utils import calculate_jackalope_kernel

def debug_favorite_alignment(appid=1240440): # Detroit
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'tags'])
    appid_to_idx = {int(aid): i for i, aid in enumerate(full_metadata['appid'])}
    idx_f = appid_to_idx[appid]
    
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    topic_dist_all = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    t_means = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy")).astype(np.float32)
    t_stds = np.load(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy")).astype(np.float32)
    
    # Pre-calculate simple masks for debug (not exhaustive)
    masks = {"tag_Detroit": full_metadata['tags'].str.contains("'Story Rich':", na=False).values}
    
    alignment_f = calculate_jackalope_kernel(
        tag_vectors, tag_norms, tag_vectors[idx_f], tag_norms[idx_f],
        sem_vectors, sem_norms, sem_vectors[idx_f], sem_norms[idx_f],
        topic_dist_all, topic_dist_all[idx_f],
        t_means, t_stds, TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA, SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA,
        precalculated_masks=None, # Will fallback to basic calc
        full_tags_series=full_metadata['tags'].fillna('')
    )
    
    print(f"Max Alignment: {np.max(alignment_f):.4f}")
    print(f"Top 10 Alignments: {sorted(alignment_f, reverse=True)[:10]}")
    print(f"Count > 0.3: {np.sum(alignment_f > 0.3)}")
    print(f"Count > 0.1: {np.sum(alignment_f > 0.1)}")
    print(f"Count > 0.05: {np.sum(alignment_f > 0.05)}")

if __name__ == "__main__":
    debug_favorite_alignment()
