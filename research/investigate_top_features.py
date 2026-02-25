import numpy as np
import json
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import W_TAG_FILE, TAG_NAMES_FILE, TOPIC_DESCRIPTIONS_FILE

def investigate_top_dimensions():
    print("Investigating Tag Dimension 0...")
    W = np.load(W_TAG_FILE)
    with open(TAG_NAMES_FILE, 'r') as f:
        tag_names = json.load(f)
        
    loadings = W[:, 0]
    top_pos_idx = np.argsort(-loadings)[:5]
    top_neg_idx = np.argsort(loadings)[:5]
    
    print("\n--- Tag Dimension 0 (Positive Loadings) ---")
    for idx in top_pos_idx:
        print("  - " + f"{tag_names[idx]:<25}" + ": " + f"{loadings[idx]:.4f}")
        
    print("\n--- Tag Dimension 0 (Negative Loadings) ---")
    for idx in top_neg_idx:
        print("  - " + f"{tag_names[idx]:<25}" + ": " + f"{loadings[idx]:.4f}")

    topic_idx = 641 - 466
    print("\nInvestigating Topic Index " + str(topic_idx) + " (Combined index 641)...")
    
    if os.path.exists(TOPIC_DESCRIPTIONS_FILE):
        with open(TOPIC_DESCRIPTIONS_FILE, 'r') as f:
            topic_desc = json.load(f)
            print("Label: " + topic_desc.get(str(topic_idx), "Unknown"))

if __name__ == "__main__":
    investigate_top_dimensions()
