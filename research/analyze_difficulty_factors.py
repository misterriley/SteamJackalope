import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
import sklearn.utils.validation
import factor_analyzer.factor_analyzer

# Monkeypatch sklearn for factor_analyzer compatibility
original_check_array = sklearn.utils.validation.check_array
def patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return original_check_array(*args, **kwargs)

sklearn.utils.validation.check_array = patched_check_array
sklearn.utils.check_array = patched_check_array
factor_analyzer.factor_analyzer.check_array = patched_check_array

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def main():
    print("Generating non-whitened tag vectors...")
    # Monkey-patch to disable whitening
    gtv.USE_TAG_WHITENING = False
    # Use temp files to avoid overwriting production artifacts
    temp_vectors = "research/temp_fa_vectors.npy"
    temp_constants = "research/temp_fa_constants.json"
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
    
    # Check for "Difficult" tag
    target_tag = "Difficult"
    if target_tag not in tag_to_idx:
        print(f"Error: '{target_tag}' tag not found.")
        return
    
    target_idx = tag_to_idx[target_tag]
    
    # Factor Analysis
    n_factors = 20
    print(f"Running Factor Analysis (n_factors={n_factors}, rotation='varimax')...")
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(vectors)
    
    loadings = fa.loadings_ # (n_features, n_factors)
    
    # Find factor with strongest loading for "Difficult"
    difficult_loadings = loadings[target_idx, :]
    best_factor_idx = np.argmax(np.abs(difficult_loadings))
    best_loading = difficult_loadings[best_factor_idx]
    
    print(f"\n'Difficult' tag loads strongest on Factor {best_factor_idx} with loading {best_loading:.4f}")
    
    # Extract weights (loadings) for this factor
    difficulty_weights = loadings[:, best_factor_idx]
    
    # Correct sign if necessary (if "Difficult" loading is negative, flip the factor)
    if best_loading < 0:
        print("Flipping factor sign to make 'Difficult' positive...")
        difficulty_weights = -difficulty_weights
        loadings[:, best_factor_idx] = -loadings[:, best_factor_idx]

    # Display top weights in this factor
    weights_with_tags = list(zip(unique_tags, difficulty_weights))
    weights_with_tags.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Positive Weights (correlated with Difficulty):")
    for tag, w in weights_with_tags[:20]:
        print(f"{tag}: {w:.4f}")
        
    print("\nTop 20 Negative Weights (anti-correlated):")
    for tag, w in weights_with_tags[-20:][::-1]:
        print(f"{tag}: {w:.4f}")
        
    # Generate Carpet Plot
    print("\nGenerating carpet plot...")
    plt.figure(figsize=(14, 12))
    sns.heatmap(loadings, cmap="vlag", center=0, xticklabels=[f"F{i}" for i in range(n_factors)])
    plt.title("Factor Loadings (Tags x Factors)")
    plt.xlabel("Factors")
    plt.ylabel("Tags (Index)")
    
    output_plot = "research/difficulty_factor_carpet_plot.png"
    plt.savefig(output_plot, dpi=300)
    print(f"Saved carpet plot to {output_plot}")
    
    # Cleanup
    if os.path.exists(temp_vectors): os.remove(temp_vectors)
    if os.path.exists(temp_constants): os.remove(temp_constants)
    if os.path.exists(temp_w_tag): os.remove(temp_w_tag)

if __name__ == "__main__":
    main()
