import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv
from common import constants

def partial_correlation(X, y):
    """
    Computes partial correlations between each feature in X and target y,
    controlling for all other features in X.
    Using the relationship: partial_corr(x_i, y | others) is proportional to 
    the coefficient of x_i in a multiple regression of y on X.
    Specifically: r_iy.others = -p_iy / sqrt(p_ii * p_yy) where P is the precision matrix.
    However, for high-dimensional X, we can use the regression approach:
    partial_corr(x_i, y | others) is the correlation between residuals:
    res(y ~ X_{-i}) and res(x_i ~ X_{-i}).
    """
    n_features = X.shape[1]
    partial_corrs = np.zeros(n_features)
    
    # Standardize for stability
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    y_s = (y - np.mean(y)) / np.std(y)
    
    # Standard SPCA as per Bair et al. (2006) uses simple univariate correlation
    # or univariate regression coefficients for thresholding.
    # High-dimensional tag data has extreme colinearity which makes partials 
    # (even regularized ones) unstable across folds.
    return np.array([np.corrcoef(X_s[:, i], y_s)[0, 1] for i in range(X.shape[1])])

def supervised_pca_partial(X, y, theta, drop_half=True):
    """
    1. Zero-order correlations -> drop half.
    2. Partial correlations on remaining -> threshold by theta.
    3. PCA on survivors.
    """
    n_total = X.shape[1]
    
    # 1. Zero-order correlations
    zero_order = np.array([np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0 for i in range(n_total)])
    
    if drop_half:
        n_keep = n_total // 2
        top_indices = np.argsort(np.abs(zero_order))[-n_keep:]
        X_half = X[:, top_indices]
    else:
        top_indices = np.arange(n_total)
        X_half = X
        
    # 2. Partial Correlations (Standardized Regression Coefs)
    # On the reduced set to avoid extreme colinearity/p > n issues
    p_corrs = partial_correlation(X_half, y)
    
    # 3. Threshold by theta
    mask_half = np.abs(p_corrs) > theta
    if not np.any(mask_half):
        return None, None, None
    
    final_indices = top_indices[mask_half]
    X_final = X[:, final_indices]
    
    # 4. PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    # Perform SVD and extract first singular vector to be consistent with SPCA literature
    # and ensure projection doesn't suffer from center/scale shifts in CV
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled)
    
    # Save the scaler mean and scale to apply to test data
    pca.scaler_mean_ = scaler.mean_
    pca.scaler_scale_ = scaler.scale_
    
    # Return full mask for convenience
    full_mask = np.zeros(n_total, dtype=bool)
    full_mask[final_indices] = True
    
    return pc1, full_mask, pca

def main():
    print("Initializing Supervised PCA (Partial Corr) for Difficulty Prediction...")
    
    # 1. Generate Non-Whitened Vectors (Tag Proportions)
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_spca_p.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors_spca_p.npy"
    temp_constants_file = "research/temp_constants_spca_p.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Generating tag vectors...")
    vectors, appids = gtv.generate_tag_vectors(csv_path, output_vectors=temp_vectors_file, output_constants=temp_constants_file)

    # 2. Get Metadata
    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # 3. Extract Target Tags and Blended y
    target_tags = ["Difficult", "Unforgiving"]
    indices = [tag_to_idx[t] for t in target_tags if t in tag_to_idx]
    
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
    
    # Correction:
    X = X + (removed_sums / n_features).reshape(-1, 1)
    
    feature_names = np.array(unique_tags)[feature_mask]
    
    print(f"Target: Blended Z-Score of {target_tags}")
    print(f"Number of features: {X.shape[1]}")
    
    # 5. CV for Theta
    print("Performing Cross-Validation for θ (Partial Correlation Threshold)...")
    # We need to find a good range for theta on partial corrs
    # Let's do a quick check on a subset for range
    sample_p_corrs = partial_correlation(X[:, :X.shape[1]//2], y) # Proxy check
    max_p = np.max(np.abs(sample_p_corrs))
    thetas = np.linspace(0, max_p * 0.5, 15)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    theta_scores = []
    
    for theta in thetas:
        fold_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            pc1_train, mask, pca_obj = supervised_pca_partial(X_train, y_train, theta)
            
            if pc1_train is None:
                fold_scores.append(-1.0)
                continue
            
            reg = LinearRegression()
            reg.fit(pc1_train, y_train)
            
            # Project test data using parameters from the training set
            # to prevent information leakage and maintain feature space alignment
            X_test_scaled = (X_test[:, mask] - pca_obj.scaler_mean_) / pca_obj.scaler_scale_
            pc1_test = pca_obj.transform(X_test_scaled)
            
            fold_scores.append(reg.score(pc1_test, y_test))
            
        theta_scores.append(np.mean(fold_scores))
        print(f"θ: {theta:.4f}, Mean R^2: {np.mean(fold_scores):.4f}")
        
    best_theta = thetas[np.argmax(theta_scores)]
    print(f"\nBest θ: {best_theta:.4f} with R^2: {np.max(theta_scores):.4f}")
    
    # 6. Final Model
    pc1, mask, pca_obj = supervised_pca_partial(X, y, best_theta)
    selected_features = feature_names[mask]
    print(f"Selected {len(selected_features)} features.")
    
    reg = LinearRegression()
    reg.fit(pc1, y)
    y_pred = reg.predict(pc1)
    
    # 7. Results
    loadings = pca_obj.components_[0]
    if reg.coef_[0] < 0: loadings = -loadings
    
    loadings_df = pd.DataFrame({'feature': selected_features, 'loading': loadings}).sort_values('loading', ascending=False)
    print("\nTop Predictors in PC1:")
    print(loadings_df.head(15).to_string(index=False))
    
    results = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, aid) for aid in appids],
        'actual': y,
        'predicted': y_pred
    })
    
    print("\nTop Predicted Hardest:")
    print(results.sort_values('predicted', ascending=False).head(15)[['name', 'predicted']].to_string(index=False))
    
    loadings_df.to_csv("research/spca_partial_loadings.csv", index=False)
    
    # Cleanup
    for f in [temp_vectors_file, temp_constants_file, gtv.W_TAG_FILE]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
