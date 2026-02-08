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

def fit_and_score(X, y, current_indices, candidate_idx):
    """
    Helper function to fit a model and calculate AIC for a single candidate.
    Used for parallel execution.
    """
    indices = current_indices + [candidate_idx]
    X_subset = X[:, indices]
    n_samples = X.shape[0]
    
    # Simple Linear Regression (using closed form for speed or sklearn)
    # Since we need RSS, sklearn is fast enough
    reg = LinearRegression()
    reg.fit(X_subset, y)
    y_pred = reg.predict(X_subset)
    
    rss = np.sum((y - y_pred) ** 2)
    k = len(indices) + 1 # +1 for intercept
    aic = calculate_aic(n_samples, rss, k)
    
    return aic, candidate_idx

def fit_and_score_removal(X, y, current_indices, remove_idx):
    """
    Helper function to calculate AIC after removing a candidate.
    """
    indices = [i for i in current_indices if i != remove_idx]
    
    # If no features left, it's the intercept-only model
    if not indices:
        n_samples = X.shape[0]
        rss = np.sum((y - np.mean(y)) ** 2)
        k = 1
        aic = calculate_aic(n_samples, rss, k)
        return aic, remove_idx
        
    X_subset = X[:, indices]
    n_samples = X.shape[0]
    
    reg = LinearRegression()
    reg.fit(X_subset, y)
    y_pred = reg.predict(X_subset)
    
    rss = np.sum((y - y_pred) ** 2)
    k = len(indices) + 1
    aic = calculate_aic(n_samples, rss, k)
    
    return aic, remove_idx

def stepwise_selection_aic_parallel(X, y, feature_names, n_jobs=24):
    """
    Perform stepwise (forward-backward) selection based on AIC using parallel processing.
    """
    n_samples, n_features = X.shape
    selected_indices = [] # Start with empty set
    
    # Initial AIC with only intercept
    rss_null = np.sum((y - np.mean(y)) ** 2)
    current_aic = calculate_aic(n_samples, rss_null, 1)
    
    print(f"Null Model AIC: {current_aic:.4f}")
    
    while True:
        changed = False
        
        # --- Forward Step ---
        candidate_indices = [i for i in range(n_features) if i not in selected_indices]
        
        if candidate_indices:
            # print(f"Forward Step: Testing {len(candidate_indices)} candidates...")
            
            # Parallel execution of candidate testing
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score)(X, y, selected_indices, idx) 
                for idx in candidate_indices
            )
            
            # Find best candidate
            results.sort(key=lambda x: x[0])
            best_candidate_aic, best_candidate_idx = results[0]
            
            if best_candidate_aic < current_aic:
                current_aic = best_candidate_aic
                selected_indices.append(best_candidate_idx)
                changed = True
                print(f"[+] Added '{feature_names[best_candidate_idx]}' | AIC: {current_aic:.4f}")
        
        # --- Backward Step ---
        if selected_indices:
            # print(f"Backward Step: Testing {len(selected_indices)} candidates for removal...")
            
            # Parallel execution of removal testing
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score_removal)(X, y, selected_indices, idx)
                for idx in selected_indices
            )
            
            # Find best removal (lowest AIC)
            results.sort(key=lambda x: x[0])
            best_removal_aic, best_removal_idx = results[0]
            
            if best_removal_aic < current_aic:
                current_aic = best_removal_aic
                selected_indices.remove(best_removal_idx)
                changed = True
                print(f"[-] Removed '{feature_names[best_removal_idx]}' | AIC: {current_aic:.4f}")
        
        if not changed:
            print("No further improvement in AIC. Convergence reached.")
            break
            
    return selected_indices, [feature_names[i] for i in selected_indices], current_aic

def main():
    print("Initializing Parallel Stepwise AIC Selection for Difficulty Prediction...")
    
    # 1. Generate Non-Whitened Vectors (Tag Proportions)
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_aic_p.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors_aic_p.npy"
    temp_constants_file = "research/temp_constants_aic_p.json"
    
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
    print("Using 24 parallel threads for AIC optimization.")
    
    # 5. Perform Stepwise Selection
    selected_indices, selected_features, final_aic = stepwise_selection_aic_parallel(X, y, feature_names, n_jobs=24)
    
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
    
    print("\nTop 10 Positive Predictors:")
    print(coef_df.head(10).to_string(index=False))
    
    print("\nTop 10 Negative Predictors:")
    print(coef_df.tail(10).to_string(index=False))
    
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
    coef_df.to_csv("research/aic_stepwise_coefficients.csv", index=False)
    results.to_csv("research/aic_stepwise_predictions.csv", index=False)
    
    # Cleanup
    try:
        if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
        if os.path.exists(temp_constants_file): os.remove(temp_constants_file)
        if os.path.exists(gtv.W_TAG_FILE): os.remove(gtv.W_TAG_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
