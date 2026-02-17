import numpy as np
import json
import os

def check_tag_magnitudes():
    with open('data/production/tag_names.json', 'r') as f:
        tag_names = json.load(f)
    
    W = np.load('data/production/w_tag.npy')
    norms = np.linalg.norm(W, axis=1)
    
    target_tags = ['6DOF', 'Web Publishing', 'Sci-fi', 'RPG', 'Robots']
    
    print("\n--- Tag Vector Norms in Whitened Space (W) ---")
    for tag in target_tags:
        if tag in tag_names:
            idx = tag_names.index(tag)
            print(f"{tag:20} : Norm = {norms[idx]:.4f}")
    
    # Show the distribution of norms
    print(f"\nGlobal Norm Stats:")
    print(f"  Mean: {np.mean(norms):.4f}")
    print(f"  Max : {np.max(norms):.4f} ({tag_names[np.argmax(norms)]})")
    print(f"  Min : {np.min(norms):.4f} ({tag_names[np.argmin(norms)]})")

if __name__ == "__main__":
    check_tag_magnitudes()
