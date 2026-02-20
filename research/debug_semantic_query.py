import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_TAG_FILE,
    W_DESC_FILE,
    W_STRUCTURAL_FILE,
    MEAN_DESC_FILE,
    MEAN_STRUCTURAL_FILE,
    METADATA_FILE,
    MODEL_NAME
)

# Load regularization constants from file to match server exactly
REG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'production', 'regularization_constants.json')
import json
if os.path.exists(REG_FILE):
    with open(REG_FILE, 'r') as f:
        reg_data = json.load(f)
        SEMANTIC_SIMILARITY_MEAN = reg_data.get("SEMANTIC_SIMILARITY_MEAN", 0.0)
        SEMANTIC_SIMILARITY_STD = reg_data.get("SEMANTIC_SIMILARITY_STD", 1.0)
else:
    SEMANTIC_SIMILARITY_MEAN = 0.0
    SEMANTIC_SIMILARITY_STD = 1.0

def debug_semantic():
    print("Loading metadata...")
    df = pd.read_parquet(METADATA_FILE)
    
    print("Loading embeddings...")
    emb_desc = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    emb_struct = np.load(EMBEDDINGS_TAG_FILE, mmap_mode='r')
    
    print("Loading transformation matrices...")
    w_desc = np.load(W_DESC_FILE)
    w_struct = np.load(W_STRUCTURAL_FILE)
    
    mean_desc = np.load(MEAN_DESC_FILE) if os.path.exists(MEAN_DESC_FILE) else np.zeros(w_desc.shape[0])
    mean_struct = np.load(MEAN_STRUCTURAL_FILE) if os.path.exists(MEAN_STRUCTURAL_FILE) else np.zeros(w_struct.shape[0])

    print(f"Loading SentenceTransformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    query = "happiness"
    print(f"Query: '{query}'")
    
    query_vec = model.encode([query])[0]
    
    # Process query like the server does
    w_desc_f32 = w_desc.astype(np.float32)
    w_struct_f32 = w_struct.astype(np.float32)
    mean_desc_f32 = mean_desc.astype(np.float32)
    mean_struct_f32 = mean_struct.astype(np.float32)

    q_desc = (query_vec - mean_desc_f32) @ w_desc_f32
    q_struct = (query_vec - mean_struct_f32) @ w_struct_f32
    
    # Normalize query vectors
    q_desc = q_desc / np.linalg.norm(q_desc)
    q_struct = q_struct / np.linalg.norm(q_struct)
    
    # Calculate similarities
    sim_desc = np.dot(emb_desc.astype(np.float32), q_desc.astype(np.float32))
    sim_struct = np.dot(emb_struct.astype(np.float32), q_struct.astype(np.float32))
    
    df['sim_desc'] = sim_desc
    df['sim_struct'] = sim_struct
    df['sim_combined'] = (sim_desc + sim_struct) / 2.0
    df['z_score'] = (df['sim_combined'] - SEMANTIC_SIMILARITY_MEAN) / SEMANTIC_SIMILARITY_STD
    
    target_appid = 1182690
    target_game = df[df['appid'] == target_appid]
    
    if not target_game.empty:
        g = target_game.iloc[0]
        print(f"\nTarget Game: {g['name']}")
        print(f"Sim Desc: {g['sim_desc']:.4f}")
        print(f"Sim Struct: {g['sim_struct']:.4f}")
        print(f"Sim Combined: {g['sim_combined']:.4f}")
        print(f"Z-Score: {g['z_score']:.4f}")
        print(f"Rank: {(df['z_score'] > g['z_score']).sum() + 1}")
    
    print("\nTop 5 Breakdown:")
    top5 = df.sort_values('z_score', ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f" - {r['name']} ({r['appid']}): Desc={r['sim_desc']:.4f}, Struct={r['sim_struct']:.4f}, Combined={r['sim_combined']:.4f}")

if __name__ == "__main__":
    debug_semantic()
