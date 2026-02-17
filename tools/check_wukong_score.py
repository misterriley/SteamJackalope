import pandas as pd
import numpy as np
import os
import json
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    QUALITY_GRID_FILE,
    TAG_GLOBAL_SCALING_FACTOR,
    DOT_PRODUCT_LAMBDA
)

def check_game_score(steamid, game_appid):
    profile_path = f"data/user_{steamid}_taste_profile.json"
    with open(profile_path, 'r') as f:
        profile = json.load(f)
        
    metadata = pd.read_parquet(METADATA_FILE)
    game_idx = metadata[metadata['appid'] == game_appid].index[0]
    
    # Load Tag Vectors
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    u = tag_vectors[game_idx].astype(np.float32)
    u_norm = np.linalg.norm(u)
    u_scaled = (u / (u_norm + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    
    vibe_vector = np.array(profile['vibe_vector'], dtype=np.float32)
    
    tag_match = np.dot(u_scaled, vibe_vector)
    
    # Metadata scores
    meta = metadata.iloc[game_idx]
    z_date = np.clip(meta['date_z'], -8, 8)
    z_pop = np.clip(meta['pop_z'], -8, 8)
    z_length = np.clip(meta['playtime_z'], -8, 8)
    z_difficulty = np.clip(meta['difficulty_z'], -8, 8)
    
    # Quality score (using discovery from profile)
    disc_pref = profile['metadata']['discovery']
    quality_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
    num_steps = quality_grid.shape[0]
    grid_index = int(round(((disc_pref - (-1.0)) / 2.0) * (num_steps - 1)))
    z_spps = np.clip(quality_grid[grid_index][game_idx], -8, 8)
    
    # Weights
    w_tag = profile['metadata']['tag_match']
    w_quality = profile['metadata']['quality']
    w_date = profile['metadata']['age']
    w_pop = profile['metadata']['popularity']
    w_length = profile['metadata']['length']
    w_difficulty = profile['metadata']['difficulty']
    
    tag_contrib = tag_match * w_tag
    quality_contrib = z_spps * w_quality
    date_contrib = z_date * w_date
    pop_contrib = z_pop * w_pop
    length_contrib = z_length * w_length
    diff_contrib = z_difficulty * w_difficulty
    
    intercept = profile['intercept']
    
    final_score = intercept + tag_contrib + quality_contrib + date_contrib + pop_contrib + length_contrib + diff_contrib
    
    print(f"Game: {meta['name']} ({game_appid})")
    print(f"Intercept: {intercept:.4f}")
    print(f"Tag Match Score: {tag_match:.4f} (Contrib: {tag_contrib:.4f}, Weight: {w_tag:.4f})")
    print(f"Quality Score: {z_spps:.4f} (Contrib: {quality_contrib:.4f}, Weight: {w_quality:.4f})")
    print(f"Date Score: {z_date:.4f} (Contrib: {date_contrib:.4f}, Weight: {w_date:.4f})")
    print(f"Pop Score: {z_pop:.4f} (Contrib: {pop_contrib:.4f}, Weight: {w_pop:.4f})")
    print(f"Length Score: {z_length:.4f} (Contrib: {length_contrib:.4f}, Weight: {w_length:.4f})")
    print(f"Difficulty Score: {z_difficulty:.4f} (Contrib: {diff_contrib:.4f}, Weight: {w_difficulty:.4f})")
    print(f"TOTAL PREDICTED RATING: {final_score:.4f}")

if __name__ == "__main__":
    check_game_score('76561198039155404', 2358720) # Black Myth: Wukong
    print("-" * 30)
    check_game_score('76561198039155404', 2466920) # Finish Report (North Star)
