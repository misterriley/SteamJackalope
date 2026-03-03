import pandas as pd
import numpy as np
import json
import os
import re

PRODUCTION_DATA_DIR = 'data/production'
df = pd.read_parquet(os.path.join(PRODUCTION_DATA_DIR, 'metadata.parquet'))

def debug_detroit_final():
    seed_name = "Detroit: Become Human"
    targets = [
        "Trinoline All Ages Version",
        "The Wreck",
        "Tale of android after the Apocalypse",
        "Beyond: Two Souls",
        "The Uncertain: Light At The End",
        "Dreamfall: The Longest Journey"
    ]
    
    idx1 = df[df['name'] == seed_name].index[0]
    
    tag_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'steam_tag_vectors.npy'), mmap_mode='r')
    tag_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'tag_vectors_norms.npy'), mmap_mode='r')
    sem_vectors = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc.npy'), mmap_mode='r')
    sem_norms = np.load(os.path.join(PRODUCTION_DATA_DIR, 'embeddings_desc_norms.npy'), mmap_mode='r')
    
    for target in targets:
        try:
            idx2 = df[df['name'] == target].index[0]
            t_sim = (np.dot(tag_vectors[idx1], tag_vectors[idx2]) / ((tag_norms[idx1] + 1.0) * (tag_norms[idx2] + 1.0))) * 11.25
            s_sim = (np.dot(sem_vectors[idx1], sem_vectors[idx2]) / ((sem_norms[idx1] + 1.0) * (sem_norms[idx2] + 1.0))) * 10.0
            print(f"{target}: Tag={t_sim:.4f}, Sem={s_sim:.4f}")
        except:
            print(f"{target} NOT FOUND")

debug_detroit_final()
