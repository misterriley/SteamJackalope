import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv
from common import constants

def calculate_bic(n, rss, k):
    """
    Calculate Bayesian Information Criterion (BIC).
    BIC = k * ln(n) + n * ln(RSS/n)
    where:
    n = number of observations
    rss = residual sum of squares
    k = number of parameters (including intercept)
    """
    if rss <= 0:
        return -np.inf
    return k * np.log(n) + n * np.log(rss / n)

def fit_and_score_bic(X, y, current_indices, candidate_idx):
    """
    Helper function to fit a model and calculate BIC for a single candidate addition.
    """
    indices = current_indices + [candidate_idx]
    X_subset = X[:, indices]
    n_samples = X.shape[0]
    
    reg = LinearRegression()
    reg.fit(X_subset, y)
    y_pred = reg.predict(X_subset)
    
    rss = np.sum((y - y_pred) ** 2)
    k = len(indices) + 1 # +1 for intercept
    bic = calculate_bic(n_samples, rss, k)
    
    return bic, candidate_idx

def fit_and_score_removal_bic(X, y, current_indices, remove_idx):
    """
    Helper function to calculate BIC after removing a candidate.
    """
    indices = [i for i in current_indices if i != remove_idx]
    n_samples = X.shape[0]
    
    if not indices:
        rss = np.sum((y - np.mean(y)) ** 2)
        k = 1
    else:
        X_subset = X[:, indices]
        reg = LinearRegression()
        reg.fit(X_subset, y)
        y_pred = reg.predict(X_subset)
        rss = np.sum((y - y_pred) ** 2)
        k = len(indices) + 1
        
    bic = calculate_bic(n_samples, rss, k)
    return bic, remove_idx

def stepwise_selection_bic_parallel(X, y, feature_names, n_jobs=24):
    """
    Perform stepwise (forward-backward) selection based on BIC using parallel processing.
    BIC is more restrictive than AIC (penalty scale ln(n) vs 2).
    """
    n_samples, n_features = X.shape
    selected_indices = []
    
    # Initial BIC with only intercept
    rss_null = np.sum((y - np.mean(y)) ** 2)
    current_bic = calculate_bic(n_samples, rss_null, 1)
    
    print(f"Null Model BIC: {current_bic:.4f}")
    
    while True:
        changed = False
        
        # --- Forward Step ---
        candidate_indices = [i for i in range(n_features) if i not in selected_indices]
        
        if candidate_indices:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score_bic)(X, y, selected_indices, idx) 
                for idx in candidate_indices
            )
            
            results.sort(key=lambda x: x[0])
            best_candidate_bic, best_candidate_idx = results[0]
            
            if best_candidate_bic < current_bic:
                current_bic = best_candidate_bic
                selected_indices.append(best_candidate_idx)
                changed = True
                print(f"[+] Added '{feature_names[best_candidate_idx]}' | BIC: {current_bic:.4f}")
        
        # --- Backward Step ---
        if selected_indices:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score_removal_bic)(X, y, selected_indices, idx)
                for idx in selected_indices
            )
            
            results.sort(key=lambda x: x[0])
            best_removal_bic, best_removal_idx = results[0]
            
            if best_removal_bic < current_bic:
                current_bic = best_removal_bic
                selected_indices.remove(best_removal_idx)
                changed = True
                print(f"[-] Removed '{feature_names[best_removal_idx]}' | BIC: {current_bic:.4f}")
        
        if not changed:
            print("No further improvement in BIC. Convergence reached.")
            break
            
    return selected_indices, [feature_names[i] for i in selected_indices], current_bic

def main():
    print("Initializing Parallel Stepwise BIC Selection for Difficulty Prediction...")
    
    # 1. Generate Non-Whitened Vectors (Tag Proportions)
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_bic_p.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors_bic_p.npy"
    temp_constants_file = "research/temp_constants_bic_p.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Generating tag vectors...")
    vectors, appids = gtv.generate_tag_vectors(
        csv_path, 
        output_vectors=temp_vectors_file, 
        output_constants=temp_constants_file
    )

    # 2. Get Metadata
    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # 3. Target Variable
    target_tags = ["Difficult", "Unforgiving"]
    indices = [tag_to_idx[t] for t in target_tags if t in tag_to_idx]
    if len(indices) != 2:
        print("Error: Target tags not found.")
        return

    y_raw = vectors[:, indices]
    y_z = (y_raw - np.mean(y_raw, axis=0)) / np.std(y_raw, axis=0)
    y = np.sum(y_z, axis=1)
    
    # 4. Prepare Features (X) with Zero-Sum Maintenance
    removed_sums = np.sum(vectors[:, indices], axis=1)
    feature_mask = np.ones(vectors.shape[1], dtype=bool)
    for idx in indices:
        feature_mask[idx] = False
    X = vectors[:, feature_mask].copy()
    n_features = X.shape[1]
    X = X + (removed_sums / n_features).reshape(-1, 1)
    feature_names = np.array(unique_tags)[feature_mask]
    
    print(f"Target: Blended Z-Score of {target_tags}")
    print(f"Number of features: {X.shape[1]}")
    print("Using 24 parallel threads for BIC optimization.")
    
    # 5. Perform Stepwise Selection
    selected_indices, selected_features, final_bic = stepwise_selection_bic_parallel(X, y, feature_names, n_jobs=24)
    
    print(f"\nFinal Model Selected {len(selected_features)} Features.")
    print(f"Final BIC: {final_bic:.4f}")
    
    # 6. Fit Final Model and Report
    X_final = X[:, selected_indices]
    reg = LinearRegression()
    reg.fit(X_final, y)
    y_pred = reg.predict(X_final)
    
    coef_df = pd.DataFrame({
        'feature': selected_features,
        'coefficient': reg.coef_
    }).sort_values('coefficient', ascending=False)
    
    print("\nModel Coefficients:")
    print(coef_df.to_string(index=False))
    
    results = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, aid) for aid in appids],
        'actual': y,
        'predicted': y_pred
    })
    
    print("\nTop Predicted Hardest Games:")
    print(results.sort_values('predicted', ascending=False).head(20)[['name', 'predicted']].to_string(index=False))
    
    # Save results
    coef_df.to_csv("research/bic_stepwise_coefficients.csv", index=False)
    results.to_csv("research/bic_stepwise_predictions.csv", index=False)
    
    # Cleanup
    try:
        if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
        if os.path.exists(temp_constants_file): os.remove(temp_constants_file)
        if os.path.exists(gtv.W_TAG_FILE): os.remove(gtv.W_TAG_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
