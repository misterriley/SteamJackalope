import numpy as np
import os

files = [
    'embeddings_desc.npy',
    'embeddings_structural.npy',
    'mean_desc.npy',
    'mean_structural.npy',
    'quality_scores_grid.npy',
    'steam_tag_vectors.npy',
    'tag_vectors_norms.npy',
    'w_desc.npy',
    'w_structural.npy',
    'w_tag.npy'
]

for f in files:
    if os.path.exists(f):
        data = np.load(f, mmap_mode='r')
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"{f}: shape={data.shape}, dtype={data.dtype}, size={size_mb:.2f} MB")
    else:
        print(f"{f}: File not found")
