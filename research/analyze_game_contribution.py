import pandas as pd
import numpy as np
import os
import sys
import json

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    METADATA_FILE, 
    ROOT_DIR,
    TAG_NORMS_FILE,
    DOT_PRODUCT_LAMBDA,
    TAG_GLOBAL_SCALING_FACTOR,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)

def analyze_game_recommendation(appid, steamid):
    profile_path = f"data/user_{steamid}_taste_profile.json"
    if not os.path.exists(profile_path):
        print(f"Error: Profile for {steamid} not found.")
        return

    with open(profile_path, 'r') as f:
        profile = json.load(f)

    weights = profile['metadata']
    vibe_unit = np.array(profile['vibe_vector'])
    tag_norm = weights.get('tag_match', 1.0)
    beta_tag = vibe_unit * tag_norm

    print(f"Loading metadata for AppID {appid}...")
    df = pd.read_parquet(METADATA_FILE)
    game = df[df['appid'] == appid].iloc[0]
    idx = df[df['appid'] == appid].index[0]

    print(f"\n--- Recommendation Analysis: {game['name']} ---")
    
    # Quality
    quality_grid = np.load(os.path.join(ROOT_DIR, "data", "production", "quality_scores_grid.npy"), mmap_mode='r')
    # Use discovery setting from profile
    disc_pref = weights.get('discovery', 0.0)
    num_steps = quality_grid.shape[0]
    step_idx = int(round(((disc_pref - (-1.0)) / 2.0) * (num_steps - 1)))
    step_idx = max(0, min(num_steps - 1, step_idx))
    
    q_val = np.clip(quality_grid[step_idx][idx], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    q_contrib = q_val * weights.get('quality', 0.0)
    
    # Metadata
    features = {
        'age': ('date_z', weights.get('age', 0.0)),
        'popularity': ('pop_z', weights.get('popularity', 0.0)),
        'length': ('playtime_z', weights.get('length', 0.0)),
        'difficulty': ('difficulty_z', weights.get('difficulty', 0.0)),
        'price': ('price_z', weights.get('price', 0.0))
    }
    
    print(f"{'Component':12} | {'Z-Score':10} | {'Weight':10} | {'Contribution':12}")
    print("-" * 50)
    print(f"{'Quality':12} | {q_val:10.4f} | {weights.get('quality', 0.0):10.4f} | {q_contrib:12.4f}")
    
    total_score = q_contrib
    for label, (col, weight) in features.items():
        val = np.clip(game[col], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
        contrib = val * weight
        total_score += contrib
        print(f"{label.capitalize():12} | {val:10.4f} | {weight:10.4f} | {contrib:12.4f}")

    # Tag Match
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    tag_norms = np.load(TAG_NORMS_FILE, mmap_mode='r')
    
    v = tag_vectors[idx].astype(np.float32)
    v_norm = tag_norms[idx].astype(np.float32)
    
    dot = np.dot(v, beta_tag)
    tag_match_raw = (dot / (v_norm + DOT_PRODUCT_LAMBDA)) * TAG_GLOBAL_SCALING_FACTOR
    tag_contrib = tag_match_raw # Weight is already in beta_tag
    total_score += tag_contrib
    
    print(f"{'Tag Match':12} | {tag_match_raw:10.4f} | {'N/A':>10} | {tag_contrib:12.4f}")
    print("-" * 50)
    print(f"{'TOTAL SCORE':40} | {total_score:12.4f}")

    # Top contributing tags (Vibe alignment)
    print("\n--- Top Tag Alignments (Vibe) ---")
    W = np.load(os.path.join(ROOT_DIR, "data", "production", "w_tag.npy"))
    # Project vibe unit back to original tag space
    vibe_original = np.dot(W, vibe_unit)
    
    with open(os.path.join(ROOT_DIR, "data", "production", "tag_names.json"), 'r') as f:
        tag_names = json.load(f)
    
    # Get tags for this game
    game_tags_str = game['tags']
    import ast
    game_tags = ast.literal_eval(game_tags_str) if isinstance(game_tags_str, str) else {}
    
    alignments = []
    for tag, count in game_tags.items():
        if tag in tag_names:
            t_idx = tag_names.index(tag)
            impact = vibe_original[t_idx]
            alignments.append((tag, impact))
            
    alignments.sort(key=lambda x: x[1], reverse=True)
    for tag, impact in alignments[:10]:
        print(f"  {tag:20}: {impact:+.4f}")

if __name__ == "__main__":
    analyze_game_recommendation(2634950, "76561198039155404")
