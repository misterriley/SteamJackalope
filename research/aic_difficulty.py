import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv
from common import constants

def calculate_aic(n, rss, k):
    """
    Calculate Akaike Information Criterion (AIC).
    AIC = 2k + n * ln(RSS/n)
    where:
    n = number of observations
    rss = residual sum of squares
    k = number of parameters (including intercept)
    """
    if rss <= 0:
        return -np.inf
    return 2 * k + n * np.log(rss / n)

def forward_selection_aic(X, y, feature_names):
    """
    Perform forward stepwise selection based on AIC.
    """
    n_samples, n_features = X.shape
    selected_indices = []
    selected_features = []
    
    # Initial AIC with only intercept
    # Intercept-only model: y_pred = mean(y)
    rss_null = np.sum((y - np.mean(y)) ** 2)
    current_aic = calculate_aic(n_samples, rss_null, 1) # k=1 for intercept
    
    print(f"Null Model AIC: {current_aic:.4f}")
    
    best_aic = current_aic
    
    while True:
        aic_candidates = []
        candidate_indices = [i for i in range(n_features) if i not in selected_indices]
        
        if not candidate_indices:
            break
            
        print(f"Testing {len(candidate_indices)} candidates...")
        
        for idx in candidate_indices:
            # Current set + candidate
            current_indices = selected_indices + [idx]
            X_current = X[:, current_indices]
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X_current, y)
            y_pred = reg.predict(X_current)
            
            rss = np.sum((y - y_pred) ** 2)
            # k = number of features + 1 (intercept)
            k = len(current_indices) + 1
            aic = calculate_aic(n_samples, rss, k)
            
            aic_candidates.append((aic, idx))
        
        # Find best candidate
        aic_candidates.sort(key=lambda x: x[0])
        best_candidate_aic, best_candidate_idx = aic_candidates[0]
        
        if best_candidate_aic < best_aic:
            best_aic = best_candidate_aic
            selected_indices.append(best_candidate_idx)
            selected_features.append(feature_names[best_candidate_idx])
            print(f"Added '{feature_names[best_candidate_idx]}' | AIC: {best_aic:.4f}")
        else:
            print("No improvement in AIC. Stopping.")
            break
            
    return selected_indices, selected_features, best_aic

def main():
    print("Initializing Stepwise AIC Selection for Difficulty Prediction...")
    
    # 1. Generate Non-Whitened Vectors (Tag Proportions)
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_aic.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors_aic.npy"
    temp_constants_file = "research/temp_constants_aic.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Generating tag vectors...")
    # Using existing logic from previous steps to generate vectors
    vectors, appids = gtv.generate_tag_vectors(
        csv_path, 
        output_vectors=temp_vectors_file, 
        output_constants=temp_constants_file
    )

    # 2. Get Metadata
    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # 3. Extract Target Tags and Blended y
    target_tags = ["Difficult", "Unforgiving"]
    indices = [tag_to_idx[t] for t in target_tags if t in tag_to_idx]
    
    if len(indices) != 2:
        print(f"Error: Could not find both target tags {target_tags}")
        return

    y_raw = vectors[:, indices]
    y_z = (y_raw - np.mean(y_raw, axis=0)) / np.std(y_raw, axis=0)
    y = np.sum(y_z, axis=1)
    
    # 4. Prepare Features (X) with Zero-Sum Maintenance
    # Sum of removed values per game
    removed_sums = np.sum(vectors[:, indices], axis=1)
    
    feature_mask = np.ones(vectors.shape[1], dtype=bool)
    for idx in indices:
        feature_mask[idx] = False
    
    X = vectors[:, feature_mask].copy()
    n_features = X.shape[1]
    
    # Correction: Distribute removed sum to maintain zero sum
    X = X + (removed_sums / n_features).reshape(-1, 1)
    
    feature_names = np.array(unique_tags)[feature_mask]
    
    print(f"Target: Blended Z-Score of {target_tags}")
    print(f"Number of features: {X.shape[1]}")
    
    # 5. Perform Stepwise Selection
    selected_indices, selected_features, final_aic = forward_selection_aic(X, y, feature_names)
    
    print(f"\nFinal Model Selected {len(selected_features)} Features.")
    print(f"Final AIC: {final_aic:.4f}")
    
    # 6. Fit Final Model and Report
    X_final = X[:, selected_indices]
    reg = LinearRegression()
    reg.fit(X_final, y)
    y_pred = reg.predict(X_final)
    
    # Coefficients
    coef_df = pd.DataFrame({
        'feature': selected_features,
        'coefficient': reg.coef_
    }).sort_values('coefficient', ascending=False)
    
    print("\nModel Coefficients:")
    print(coef_df.to_string(index=False))
    
    # Predictions
    results = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, aid) for aid in appids],
        'actual': y,
        'predicted': y_pred
    })
    
    print("\nTop Predicted Hardest Games:")
    print(results.sort_values('predicted', ascending=False).head(20)[['name', 'predicted']].to_string(index=False))
    
    # Save results
    coef_df.to_csv("research/aic_coefficients.csv", index=False)
    results.to_csv("research/aic_predictions.csv", index=False)
    
    # Cleanup
    try:
        if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
        if os.path.exists(temp_constants_file): os.remove(temp_constants_file)
        if os.path.exists(gtv.W_TAG_FILE): os.remove(gtv.W_TAG_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
