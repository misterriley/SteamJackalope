import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_DESC_NORMS_FILE,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    SEMANTIC_DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR
)

def check_feature_scales(user_id):
    ground_truth_path = f"data/user_{user_id}_ground_truth.csv"
    df_gt = pd.read_csv(ground_truth_path)
    df = df_gt[df_gt['ignore'] == False].dropna(subset=['actual_rating'])
    user_appids = df['appid'].values
    
    full_metadata = pd.read_parquet(METADATA_FILE, columns=['appid'])
    appid_to_idx = {appid: idx for idx, appid in enumerate(full_metadata['appid'])}
    user_indices = [appid_to_idx[aid] for aid in user_appids if aid in appid_to_idx]
    
    # Tag Scales
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    full_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    user_tag_norms = full_norms[user_indices].reshape(-1, 1).astype(np.float32)
    user_tag_features = tag_vectors[user_indices].astype(np.float32)
    user_tag_features_scaled = (user_tag_features / (user_tag_norms + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    # Semantic Scales
    sem_vectors = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    sem_norms = np.load(EMBEDDINGS_DESC_NORMS_FILE, mmap_mode='r')
    user_sem_norms = sem_norms[user_indices].reshape(-1, 1).astype(np.float32)
    user_sem_features = sem_vectors[user_indices].astype(np.float32)
    user_sem_features_scaled = (user_sem_features / (user_sem_norms + SEMANTIC_DOT_PRODUCT_LAMBDA)) * SEMANTIC_GLOBAL_SCALING_FACTOR
    
    print(f"Tag Features (Scaled): Mean Std = {np.std(user_tag_features_scaled, axis=0).mean():.6f}")
    print(f"Sem Features (Scaled): Mean Std = {np.std(user_sem_features_scaled, axis=0).mean():.6f}")
    
    # Suggest a scaling factor to match tag variance
    ratio = np.std(user_tag_features_scaled, axis=0).mean() / np.std(user_sem_features_scaled, axis=0).mean()
    print(f"Suggested SEMANTIC_GLOBAL_SCALING_FACTOR: {ratio:.4f}")

if __name__ == "__main__":
    check_feature_scales("76561198039155404")
