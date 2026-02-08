import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def main():
    print("Initializing Leverage and Influence Analysis...")
    
    # 1. Load Data and Metadata
    csv_path = "data/pipeline_games_clean.csv"
    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # 2. Reconstruct X and y (using the BIC selected features)
    # Load coefficients to get selected features
    coef_df = pd.read_csv("research/bic_stepwise_coefficients.csv")
    selected_features = coef_df['feature'].tolist()
    
    # Generate vectors (using temp files)
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_lev.npy"
    vectors, appids = gtv.generate_tag_vectors(csv_path, output_vectors="research/temp_vectors_lev.npy")
    
    # Target Variable (Blended)
    target_tags = ["Difficult", "Unforgiving"]
    indices = [tag_to_idx[t] for t in target_tags]
    y_raw = vectors[:, indices]
    y_z = (y_raw - np.mean(y_raw, axis=0)) / np.std(y_raw, axis=0)
    y = np.sum(y_z, axis=1)
    
    # Features (Selected by BIC)
    removed_sums = np.sum(vectors[:, indices], axis=1)
    feature_mask = np.ones(vectors.shape[1], dtype=bool)
    for idx in indices: feature_mask[idx] = False
    
    X_full = vectors[:, feature_mask].copy()
    n_features_full = X_full.shape[1]
    X_full = X_full + (removed_sums / n_features_full).reshape(-1, 1)
    
    # Subset to selected features
    selected_indices_in_X = [np.where(np.array(unique_tags)[feature_mask] == name)[0][0] for name in selected_features]
    X = X_full[:, selected_indices_in_X]
    
    print(f"Analyzing {X.shape[0]} games with {X.shape[1]} features...")
    
    # 3. Fit Model and Calculate Diagnostics
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    
    # Leverage (Hat Matrix diagonal)
    # H = X(X'X)^-1X'
    # leverage_i = x_i'(X'X)^-1x_i
    # For large datasets, we can use the SVD approach: H = U U' where X = UDV'
    from scipy.linalg import svd
    # Center X for leverage calculation
    X_centered = X - np.mean(X, axis=0)
    U, _, _ = svd(X_centered, full_matrices=False)
    leverage = np.sum(U**2, axis=1)
    
    # Cook's Distance (Influence)
    # D_i = (res_i^2 / (p * MSE)) * (lev_i / (1 - lev_i)^2)
    p = X.shape[1]
    n = X.shape[0]
    mse = np.sum(residuals**2) / (n - p)
    cooks_d = (residuals**2 / (p * mse)) * (leverage / (1 - leverage)**2)
    
    # 4. Results DataFrame
    diagnostics = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, aid) for aid in appids],
        'actual': y,
        'predicted': y_pred,
        'residual': residuals,
        'leverage': leverage,
        'cooks_d': cooks_d
    })
    
    # 5. Thresholds
    # Leverage threshold: 2 * (p+1) / n
    lev_thresh = 2 * (p + 1) / n
    # Cook's D threshold: 4 / n
    cook_thresh = 4 / n
    
    print(f"Leverage Threshold: {lev_thresh:.6f}")
    print(f"Cook's D Threshold: {cook_thresh:.6f}")
    
    # 6. Recommendations for Exclusion
    high_influence = diagnostics[diagnostics['cooks_d'] > cook_thresh].sort_values('cooks_d', ascending=False)
    high_leverage = diagnostics[diagnostics['leverage'] > lev_thresh].sort_values('leverage', ascending=False)
    
    print("\nTop 20 Most Influential Games (High Cook's D):")
    print(high_influence.head(20)[['name', 'actual', 'predicted', 'cooks_d']].to_string(index=False))
    
    print("\nTop 20 High Leverage Games (Outliers in Tag Space):")
    print(high_leverage.head(20)[['name', 'leverage']].to_string(index=False))
    
    # Summary of exclusions
    exclude_list = high_influence['appid'].head(100).tolist() # Recommend top 100 for a start
    print(f"\nRecommended for exclusion (Top 100 by influence): {len(exclude_list)} games.")
    
    # 7. Visualization: Influence Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(leverage, residuals, c=cooks_d, cmap='viridis', alpha=0.5)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(lev_thresh, color='red', linestyle='--', label='Lev Thresh')
    plt.colorbar(label="Cook's D")
    plt.xlabel('Leverage')
    plt.ylabel('Residuals')
    plt.title('Influence Plot (Leverage vs Residuals)')
    plt.savefig('research/leverage_analysis.png')
    
    # Save diagnostics
    diagnostics.to_csv("research/influence_diagnostics.csv", index=False)
    
    # Cleanup
    for f in ["research/temp_w_tag_lev.npy", "research/temp_vectors_lev.npy"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
