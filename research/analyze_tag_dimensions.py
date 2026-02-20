import numpy as np
import json
import os
import sys
import pandas as pd

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import ROOT_DIR

def load_tag_counts():
    counts_path = os.path.join(ROOT_DIR, "research", "tag_counts.csv")
    if not os.path.exists(counts_path):
        return {}
    df = pd.read_csv(counts_path)
    return dict(zip(df['tag'], df['count']))

def get_refined_description(dim_idx, pos_tags, neg_tags, tag_counts):
    # Manual refinements for notable dimensions based on the analysis
    # Ensure refinements don't use rare tags
    refinements = {
        0: "Tactical Combat vs Adult Themes",
        1: "Political Strategy vs Fast Action",
        2: "Vehicle Sim vs Turn-Based Tactics",
        3: "Casual Logic vs Horror Survival",
        4: "Adult Content vs Nature Exploration",
        5: "Atmospheric Mystery vs Casual Adventure",
        6: "Anime Adult vs Strategy Puzzle",
        9: "Automation Economy vs Multiplayer Casual",
        10: "Action Roguelike vs 2D Platformer",
        11: "Political Narrative vs Adult Themes",
        12: "RPGMaker Logic vs Precision Platforming",
        19: "3D Strategy vs 2D Casual",
        29: "Programming Sim vs Retro Arcade",
        45: "Anime Fantasy vs Educational Sim",
        59: "Relaxing VR vs Sports Shooter",
    }
    
    if dim_idx in refinements:
        return refinements[dim_idx]
    
    # Generic refinement logic: Pick top tags that aren't too rare
    # Threshold for "rare": < 500 games (approx 0.3% of games with tags)
    threshold = 500
    
    common_pos = [t for t in pos_tags if tag_counts.get(t, 0) >= threshold]
    common_neg = [t for t in neg_tags if tag_counts.get(t, 0) >= threshold]
    
    # Fallback to whatever we have if all are rare
    if not common_pos: common_pos = pos_tags
    if not common_neg: common_neg = neg_tags
    
    t1 = common_pos[0]
    t2 = common_pos[1] if len(common_pos) > 1 else ""
    
    desc = f"{t1} and {t2}" if t2 else t1
    
    if len(desc.split()) > 5:
        desc = " ".join(desc.split()[:5])
        
    return desc

def analyze_tag_dimensions():
    w_tag_path = os.path.join(ROOT_DIR, "data", "production", "w_tag.npy")
    tag_names_path = os.path.join(ROOT_DIR, "data", "production", "tag_names.json")
    output_desc_path = os.path.join(ROOT_DIR, "data", "production", "tag_dimension_descriptions.json")

    if not os.path.exists(w_tag_path) or not os.path.exists(tag_names_path):
        print("Missing required files.")
        return

    W = np.load(w_tag_path).astype(np.float32)
    with open(tag_names_path, 'r') as f:
        tag_names = json.load(f)

    tag_counts = load_tag_counts()

    num_tags, num_dims = W.shape
    descriptions = {}

    for i in range(num_dims):
        weights = W[:, i]
        # Get more tags initially to filter rare ones for description
        top_pos_idx = np.argsort(weights)[-20:][::-1]
        top_neg_idx = np.argsort(weights)[:20]
        
        pos_tags = [tag_names[idx] for idx in top_pos_idx]
        neg_tags = [tag_names[idx] for idx in top_neg_idx]
        
        desc = get_refined_description(i, pos_tags, neg_tags, tag_counts)
            
        descriptions[str(i)] = {
            "description": desc,
            "top_positive": pos_tags[:10], # Save top 10 for the sanity check later
            "top_negative": neg_tags[:10]
        }

    with open(output_desc_path, 'w') as f:
        json.dump(descriptions, f, indent=4)
    
    print(f"Saved refined descriptions to {output_desc_path}")

if __name__ == "__main__":
    analyze_tag_dimensions()
