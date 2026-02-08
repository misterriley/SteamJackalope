import sys
import os
import numpy as np
import pandas as pd
import scipy.stats

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def main():
    print("Generating non-whitened tag vectors...")
    # Monkey-patch to disable whitening
    gtv.USE_TAG_WHITENING = False
    # Use temp files
    temp_vectors = "research/temp_corr_vectors.npy"
    temp_constants = "research/temp_corr_constants.json"
    temp_w_tag = "research/temp_w_tag.npy"
    gtv.W_TAG_FILE = temp_w_tag
    
    csv_path = "data/pipeline_games_clean.csv"
    
    try:
        vectors, appids = gtv.generate_tag_vectors(
            csv_path, 
            output_vectors=temp_vectors, 
            output_constants=temp_constants
        )
    except Exception as e:
        print(f"Error generating vectors: {e}")
        return

    # Get tag names
    print("Parsing tag names...")
    df = gtv.load_data(csv_path)
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    target_tag = "Difficult"
    if target_tag not in tag_to_idx:
        print(f"Error: '{target_tag}' tag not found.")
        return
    
    target_idx = tag_to_idx[target_tag]
    target_vector = vectors[:, target_idx]
    
    print(f"Calculating correlations with '{target_tag}'...")
    
    correlations = []
    
    for i, tag in enumerate(unique_tags):
        if i == target_idx:
            continue
            
        other_vector = vectors[:, i]
        
        # Calculate Pearson correlation
        # Using numpy for speed
        # corr = cov(x,y) / (std(x)*std(y))
        # But we can use np.corrcoef
        # Note: vectors are not necessarily centered if they are just outputs of generate_tag_vectors 
        # (though generate_tag_vectors does centering in apply_tag_transform).
        # Let's use scipy.stats.pearsonr for p-values too if needed, or just np.corrcoef for speed.
        
        r = np.corrcoef(target_vector, other_vector)[0, 1]
        correlations.append((tag, r))
        
    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Positively Correlated Tags:")
    for tag, r in correlations[:20]:
        print(f"{tag}: {r:.4f}")
        
    print("\nTop 20 Negatively Correlated Tags:")
    for tag, r in correlations[-20:][::-1]:
        print(f"{tag}: {r:.4f}")

    # Cleanup
    if os.path.exists(temp_vectors): os.remove(temp_vectors)
    if os.path.exists(temp_constants): os.remove(temp_constants)
    if os.path.exists(temp_w_tag): os.remove(temp_w_tag)

if __name__ == "__main__":
    main()
