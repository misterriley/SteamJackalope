import pandas as pd
import numpy as np
import os
import json
import ast
import sys
from tqdm import tqdm

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_VECTORS_FILE, 
    W_TAG_FILE,
    ROOT_DIR
)

def get_tag_names(csv_path="data/pipeline_games_clean.csv"):
    print(f"Extracting tag names from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['tags'])
    global_tags = set()
    for tag_str in tqdm(df['tags'], desc="Parsing tags"):
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
            continue
        try:
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                global_tags.update(tags_dict.keys())
        except: continue
    return sorted(list(global_tags))

def reconstruct_ideal_game(taste_profile_path, csv_path="data/pipeline_games_clean.csv"):
    """
    Reconstructs the tag profile of an 'Ideal Game' by projecting the 
    latent taste vector back to the tag space using the pseudo-inverse.
    """
    # 1. Load Taste Profile
    print(f"Loading taste profile from {taste_profile_path}...")
    with open(taste_profile_path, 'r') as f:
        profile = json.load(f)
    
    # Beta_white is the direction in whitened space that the user likes
    beta_white = np.array(profile['vibe_vector']) # Shape (128,)
    
    # 2. Load Whitening Matrix W (Num_Tags, 128)
    print(f"Loading whitening matrix from {W_TAG_FILE}...")
    W = np.load(W_TAG_FILE).astype(np.float32)
    
    # 3. Compute Pseudo-inverse W_pinv (128, Num_Tags)
    print("Calculating pseudo-inverse of the whitening map...")
    W_pinv = np.linalg.pinv(W)
    
    # 4. Project the vector back to CLR tag space
    # This represents the "Log-Proportions" of tags in the ideal game
    ideal_clr = np.dot(beta_white, W_pinv)
    
    # 5. Invert CLR (Exponential) to get relative proportions
    # We subtract the mean for numerical stability before exp
    ideal_clr_centered = ideal_clr - np.mean(ideal_clr)
    ideal_proportions = np.exp(ideal_clr_centered)
    
    # Normalize to percentages
    ideal_proportions = (ideal_proportions / np.sum(ideal_proportions)) * 100
    
    # 6. Map to Names
    tag_names = get_tag_names(csv_path)
    
    if len(tag_names) != len(ideal_proportions):
        print(f"Warning: Tag count mismatch! Names: {len(tag_names)}, Coefficients: {len(ideal_proportions)}")
        min_len = min(len(tag_names), len(ideal_proportions))
        tag_names = tag_names[:min_len]
        ideal_proportions = ideal_proportions[:min_len]

    results = pd.DataFrame({
        'tag': tag_names,
        'probability': ideal_proportions
    })
    
    # Sort
    top_tags = results.sort_values('probability', ascending=False).head(30)
    
    print("\n" + "="*50)
    print("RECONSTRUCTED TAG PROFILE: THE 'PERFECT GAME'")
    print("="*50)
    print(f"{'Steam Tag':25} | {'Strength'}")
    print("-" * 50)
    for _, row in top_tags.iterrows():
        print(f"{row['tag']:25} | {row['probability']:.4f}%")
        
    print("\n" + "="*50)
    print("Note: These are the tag proportions that represent the")
    print("'mathematical ideal' matching your solved taste vector.")
    print("="*50)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/reconstruct_perfect_game.py <taste_profile_json>")
        sys.exit(1)
        
    profile_path = sys.argv[1]
    reconstruct_ideal_game(profile_path)
