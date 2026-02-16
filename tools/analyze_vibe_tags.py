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
    """
    Recreates the sorted list of unique tags used during vector generation.
    """
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
        except:
            continue
            
    return sorted(list(global_tags))

def analyze_vibe_components(taste_profile_path, csv_path="data/pipeline_games_clean.csv"):
    """
    Maps 128-dim vibe coefficients back to original Steam tags.
    """
    # 1. Load Taste Profile
    print(f"Loading taste profile from {taste_profile_path}...")
    with open(taste_profile_path, 'r') as f:
        profile = json.load(f)
    
    beta_vibe = np.array(profile['vibe_vector']) # Shape (128,)
    
    # 2. Load Whitening Matrix W
    # W was used as: whitened = clr_vectors @ W
    # So W has shape (Num_Tags, 128)
    print(f"Loading whitening matrix from {W_TAG_FILE}...")
    W = np.load(W_TAG_FILE).astype(np.float32)
    
    # 3. Project coefficients back to tag space
    # beta_clr = W @ beta_vibe
    # Resulting shape: (Num_Tags,)
    beta_clr = np.dot(W, beta_vibe)
    
    # 4. Get Tag Names
    tag_names = get_tag_names(csv_path)
    
    if len(tag_names) != len(beta_clr):
        print(f"Warning: Tag count mismatch! Names: {len(tag_names)}, Coefficients: {len(beta_clr)}")
        # This can happen if the production dataset was updated since w_tag.npy was saved.
        min_len = min(len(tag_names), len(beta_clr))
        tag_names = tag_names[:min_len]
        beta_clr = beta_clr[:min_len]

    # 5. Create Results DataFrame
    results = pd.DataFrame({
        'tag': tag_names,
        'coefficient': beta_clr
    })
    
    # Sort
    top_pos = results.sort_values('coefficient', ascending=False).head(20)
    top_neg = results.sort_values('coefficient', ascending=True).head(20)
    
    print("\n" + "="*40)
    print("TOP POSITIVE TAGS (Your Favorites)")
    print("="*40)
    for _, row in top_pos.iterrows():
        print(f"{row['tag']:25} : {row['coefficient']:+.4f}")
        
    print("\n" + "="*40)
    print("TOP NEGATIVE TAGS (What you avoid)")
    print("="*40)
    for _, row in top_neg.iterrows():
        print(f"{row['tag']:25} : {row['coefficient']:+.4f}")
        
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_vibe_tags.py <taste_profile_json>")
        sys.exit(1)
        
    profile_path = sys.argv[1]
    analyze_vibe_components(profile_path)
