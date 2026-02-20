import json
import numpy as np
import os
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.constants import W_TAG_FILE

def interpret_vibe_vector(user_id, num_dimensions=5, top_tags_per_dimension=5):
    profile_path = f'data/user_{user_id}_taste_profile.json'
    tag_names_path = 'data/production/tag_names.json'
    w_tag_path = W_TAG_FILE

    if not os.path.exists(profile_path):
        return f"Error: User profile not found at {profile_path}"
    if not os.path.exists(tag_names_path):
        return f"Error: Tag names not found at {tag_names_path}"
    if not os.path.exists(w_tag_path):
        return f"Error: Whitening matrix not found at {w_tag_path}"

    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    with open(tag_names_path, 'r') as f:
        tag_names = json.load(f)
        
    w_tag = np.load(w_tag_path)

    vibe_vector = np.array(profile['vibe_vector'])
    
    # Identify the indices of the largest absolute values in the vibe_vector
    # These are the most influential whitened dimensions for the user
    influential_whitened_dims_indices = np.argsort(np.abs(vibe_vector))[::-1][:num_dimensions]
    
    results = []
    results.append(f"Analyzing user {user_id}'s Vibe Vector:")
    results.append(f"  Total non-zero whitened dimensions: {np.sum(vibe_vector != 0)}")
    
    if np.sum(vibe_vector != 0) == 0:
        results.append("  All whitened tag dimensions are zero. No specific tag preferences found.")
        return "\n".join(results)

    for dim_idx in influential_whitened_dims_indices:
        vibe_val = vibe_vector[dim_idx]
        
        original_tag_contributions_to_dim = w_tag[:, dim_idx]
        
        weighted_contributions = original_tag_contributions_to_dim * vibe_val
        
        sorted_indices = np.argsort(np.abs(weighted_contributions))[::-1]
        
        results.append(f"\nDimension {dim_idx} (Whitened Vibe: {vibe_val:+.4f}):")
        results.append("  Most Influential Original Tags:")
        
        for i in range(top_tags_per_dimension):
            tag_orig_idx = sorted_indices[i]
            tag_name = tag_names[tag_orig_idx]
            contribution = weighted_contributions[tag_orig_idx]
            
            preference_type = "PREFERS" if contribution > 0 else "AVOIDS"
            
            results.append(f"    - {tag_name:25s} | Contribution: {contribution:+.4f} ({preference_type})")
            
    return "\n".join(results)

if __name__ == "__main__":
    user_id = "76561198039155404"
    print(interpret_vibe_vector(user_id))
