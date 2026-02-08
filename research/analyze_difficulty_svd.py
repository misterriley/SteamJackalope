import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def main():
    print("Loading data...")
    csv_path = "data/pipeline_games_clean.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = gtv.load_data(csv_path)
    
    # 1. Construct Game-by-Tag matrix (Counts)
    print("Parsing tags to build Count Matrix...")
    sparse_counts, tag_to_idx, unique_tags, appids = gtv.parse_tags(df)
    
    # Check for "Difficult" tag
    target_tag = "Difficult"
    if target_tag not in tag_to_idx:
        print(f"Error: '{target_tag}' tag not found.")
        return
    target_idx = tag_to_idx[target_tag]
    
    # 2. Apply TF-IDF Normalization
    print("Applying TF-IDF Normalization...")
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(sparse_counts)
    
    # 3. Perform SVD
    n_components = 50
    print(f"Performing Truncated SVD (n_components={n_components})...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf_matrix)
    
    # Components: (n_components, n_features)
    components = svd.components_
    
    # 4. Find component with highest loading for "Difficult"
    # We look at the magnitude of the loading for the target tag
    difficult_loadings = components[:, target_idx]
    best_component_idx = np.argmax(np.abs(difficult_loadings))
    best_loading = difficult_loadings[best_component_idx]
    
    print(f"\n'Difficult' tag loads strongest on Component {best_component_idx} with loading {best_loading:.4f}")
    
    # Extract weights for this component
    difficulty_weights = components[best_component_idx]
    
    # Correct sign if necessary
    if best_loading < 0:
        print("Flipping component sign to make 'Difficult' positive...")
        difficulty_weights = -difficulty_weights
        
    # Display results
    weights_with_tags = list(zip(unique_tags, difficulty_weights))
    weights_with_tags.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Positive Weights (Harder):")
    for tag, w in weights_with_tags[:20]:
        print(f"{tag}: {w:.4f}")
        
    print("\nTop 20 Negative Weights (Easier/Different):")
    for tag, w in weights_with_tags[-20:][::-1]:
        print(f"{tag}: {w:.4f}")
        
    # Save weights to CSV for future use/inspection
    output_weights = "research/svd_difficulty_weights.csv"
    pd.DataFrame(weights_with_tags, columns=["Tag", "Weight"]).to_csv(output_weights, index=False)
    print(f"\nSaved weights to {output_weights}")

    # Generate Carpet Plot of Top Components
    print("\nGenerating carpet plot of top components...")
    plt.figure(figsize=(14, 12))
    # We plot the first 20 components to keep it readable
    n_plot = 20
    sns.heatmap(components[:n_plot, :].T, cmap="vlag", center=0, xticklabels=[f"C{i}" for i in range(n_plot)])
    plt.title("SVD Component Loadings (Tags x Components)")
    plt.xlabel("Components")
    plt.ylabel("Tags (Index)")
    plt.tight_layout()
    
    output_plot = "research/svd_difficulty_carpet_plot.png"
    plt.savefig(output_plot, dpi=300)
    print(f"Saved carpet plot to {output_plot}")

if __name__ == "__main__":
    main()
