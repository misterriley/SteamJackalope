import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

# --- Refinement Constraints ---
EXCLUDED_TAGS = ["Football", "VR Only", "Football:", "Football\:", "Tile-Matching", "Soccer", "Tennis", "Basketball", "Hockey"]
# We will use a more lenient threshold or keep it and see what happens with exclusions
MIN_CORRELATION = 0.1 

def calculate_bic(n, rss, k):
    if rss <= 0: return -np.inf
    return k * np.log(n) + n * np.log(rss / n)

def fit_and_score_bic(X, y, current_indices, candidate_idx):
    indices = current_indices + [candidate_idx]
    X_subset = X[:, indices]
    reg = LinearRegression()
    reg.fit(X_subset, y)
    rss = np.sum((y - reg.predict(X_subset)) ** 2)
    return calculate_bic(X.shape[0], rss, len(indices) + 1), candidate_idx

def fit_and_score_removal_bic(X, y, current_indices, remove_idx):
    indices = [i for i in current_indices if i != remove_idx]
    if not indices:
        rss = np.sum((y - np.mean(y)) ** 2)
        k = 1
    else:
        X_subset = X[:, indices]
        reg = LinearRegression()
        reg.fit(X_subset, y)
        rss = np.sum((y - reg.predict(X_subset)) ** 2)
        k = len(indices) + 1
    return calculate_bic(X.shape[0], rss, k), remove_idx

def main():
    print(f"Initializing Stepwise BIC Refinement...")
    print(f"Initial Exclusions: {EXCLUDED_TAGS}")
    print(f"Constraint: |Zero-Order Correlation| >= {MIN_CORRELATION}")
    
    # 1. Load Data
    csv_path = "data/pipeline_games_clean.csv"
    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # Generate vectors
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_refine.npy"
    vectors, appids = gtv.generate_tag_vectors(csv_path, output_vectors="research/temp_vectors_refine.npy")
    
    # 2. Setup Target Variable (y)
    target_tags = ["Difficult", "Unforgiving"]
    target_indices = [tag_to_idx[t] for t in target_tags]
    y_raw = vectors[:, target_indices]
    y_z = (y_raw - np.mean(y_raw, axis=0)) / np.std(y_raw, axis=0)
    y = np.sum(y_z, axis=1)
    
    # 3. Setup Features (X) with Zero-Sum Maintenance
    removed_sums = np.sum(vectors[:, target_indices], axis=1)
    feature_mask = np.ones(vectors.shape[1], dtype=bool)
    for idx in target_indices: feature_mask[idx] = False
    
    X_full = vectors[:, feature_mask].copy()
    n_features_full = X_full.shape[1]
    X_full = X_full + (removed_sums / n_features_full).reshape(-1, 1)
    feature_names = np.array(unique_tags)[feature_mask]
    
    # 4. Correlation Constraint
    print("Calculating zero-order correlations...")
    correlations = np.array([np.corrcoef(X_full[:, i], y)[0, 1] for i in range(X_full.shape[1])])
    correlation_mask = np.abs(correlations) >= MIN_CORRELATION
    print(f"Features passing correlation threshold (>= {MIN_CORRELATION}): {correlation_mask.sum()}")
    
    # Final exclusion list
    exclude_indices = []
    for i, name in enumerate(feature_names):
        if not correlation_mask[i] or name in EXCLUDED_TAGS or any(ex in name for ex in ["Football", "Soccer"]):
            exclude_indices.append(i)
    
    print(f"Total features excluded: {len(exclude_indices)}")
    
    # 5. Iterative BIC Selection
    selected_indices = []
    current_bic = calculate_bic(X_full.shape[0], np.sum((y - np.mean(y))**2), 1)
    print(f"Null Model BIC: {current_bic:.4f}")
    
    n_jobs = 24
    n_features = X_full.shape[1]
    
    while True:
        changed = False
        
        # --- Forward Step ---
        candidate_indices = [i for i in range(n_features) if i not in selected_indices and i not in exclude_indices]
        if candidate_indices:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score_bic)(X_full, y, selected_indices, idx) 
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
                delayed(fit_and_score_removal_bic)(X_full, y, selected_indices, idx)
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
            print("Convergence reached.")
            break
            
    # 6. Finalize and Save
    selected_features = [feature_names[i] for i in selected_indices]
    X_final = X_full[:, selected_indices]
    reg = LinearRegression()
    reg.fit(X_final, y)
    
    coef_df = pd.DataFrame({
        'feature': selected_features,
        'coefficient': reg.coef_
    }).sort_values('coefficient', ascending=False)
    
    coef_df.to_csv("research/bic_stepwise_coefficients.csv", index=False)
    
    results = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, aid) for aid in appids],
        'actual': y,
        'predicted': reg.predict(X_final)
    })
    results.to_csv("research/bic_stepwise_predictions.csv", index=False)
    
    print(f"\nFinal model selected {len(selected_features)} features.")
    print("Top Predictors:")
    print(coef_df.to_string(index=False))
    
    # Cleanup
    for f in ["research/temp_w_tag_refine.npy", "research/temp_vectors_refine.npy"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
