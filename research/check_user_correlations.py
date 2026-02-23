import pandas as pd
import numpy as np
import os
import json
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.constants import (
    TAG_VECTORS_FILE, TAG_NORMS_FILE, METADATA_FILE, 
    REGULARIZATION_FILE, QUALITY_GRID_FILE
)

# Load user data
uid = "76561198039155404"
df = pd.read_csv(f'data/user_{uid}_ground_truth.csv')
appids = df['appid'].values
y = df['actual_rating'].values

# Load production artifacts
full_metadata = pd.read_parquet(METADATA_FILE)
appid_to_idx = {aid: i for i, aid in enumerate(full_metadata['appid'].values)}
user_indices = [appid_to_idx[aid] for aid in appids if aid in appid_to_idx]
y = y[[i for i, aid in enumerate(appids) if aid in appid_to_idx]]

# Get features
quality_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
q_global = quality_grid[20][user_indices]

meta_cols = ['date_z', 'pop_z', 'playtime_z', 'difficulty_z', 'price_z']
user_meta_features = full_metadata.iloc[user_indices][meta_cols].values

# Get Tags
tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
user_tag_features_raw = tag_vectors[user_indices].astype(np.float32)

# Load constants
reg = json.load(open(REGULARIZATION_FILE))
lambda_val = reg.get("TAG_DOT_PRODUCT_LAMBDA", 0.0)
scale_val = reg.get("TAG_GLOBAL_SCALING_FACTOR", 1.0)

full_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
user_tag_norms = full_norms[user_indices].reshape(-1, 1).astype(np.float32)

user_tag_features_scaled = (user_tag_features_raw / (user_tag_norms + lambda_val)) * scale_val

from scipy.stats import pearsonr
print(f"User ID: {uid}")
print(f"Library Size: {len(y)}")
print("\n--- Correlation with Target (actual_rating) ---")
print(f"Quality:    {pearsonr(q_global, y)[0]:.4f}")
print(f"Age:        {pearsonr(user_meta_features[:,0], y)[0]:.4f}")
print(f"Popularity: {pearsonr(user_meta_features[:,1], y)[0]:.4f}")
print(f"Playtime:   {pearsonr(user_meta_features[:,2], y)[0]:.4f}")
print(f"Difficulty: {pearsonr(user_meta_features[:,3], y)[0]:.4f}")
print(f"Price:      {pearsonr(user_meta_features[:,4], y)[0]:.4f}")

tag_corrs = [pearsonr(user_tag_features_scaled[:, i], y)[0] for i in range(user_tag_features_scaled.shape[1])]
tag_corrs = np.nan_to_num(tag_corrs)
abs_tag_corrs = np.abs(tag_corrs)
top_10_indices = np.argsort(-abs_tag_corrs)[:10]

print("\nTop 10 Tag (Whitened) Correlations:")
for idx in top_10_indices:
    print(f"  Feature {idx:3d}: {tag_corrs[idx]:+.4f}")

print("\n--- Collinearity Check ---")
print(f"Corr(Quality, Age):        {pearsonr(q_global, user_meta_features[:,0])[0]:.4f}")
print(f"Corr(Quality, Pop):        {pearsonr(q_global, user_meta_features[:,1])[0]:.4f}")

print("\n--- Variance Check ---")
print(f"Var(Quality):    {np.var(q_global):.4f}")
print(f"Var(Metadata):   {np.var(user_meta_features, axis=0)}")
print(f"Var(Tags):       {np.var(user_tag_features_scaled, axis=0).mean():.4f}")
