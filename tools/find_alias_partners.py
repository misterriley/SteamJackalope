import numpy as np
import json
import os

def find_alias_partners(target_tag, top_n=5):
    with open('data/production/tag_names.json', 'r') as f:
        tag_names = json.load(f)
    
    W = np.load('data/production/w_tag.npy')
    
    if target_tag not in tag_names:
        print(f"Tag '{target_tag}' not found in master list.")
        return
    
    idx = tag_names.index(target_tag)
    target_vec = W[idx]
    
    # Calculate similarities to all other tags
    norms = np.linalg.norm(W, axis=1)
    norms[norms == 0] = 1.0
    
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        print(f"Tag '{target_tag}' has a zero vector in W.")
        return

    similarities = np.dot(W, target_vec) / (norms * target_norm)
    
    # Sort and find top partners
    sorted_indices = np.argsort(-similarities)
    
    print(f"\n--- Top Alias Partners for '{target_tag}' ---")
    for i in range(1, top_n + 1):
        partner_idx = sorted_indices[i]
        print(f"{tag_names[partner_idx]:20} : Similarity = {similarities[partner_idx]:.4f}")

if __name__ == "__main__":
    find_alias_partners('6DOF')
    find_alias_partners('Web Publishing')
