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

def supervised_pca(X, y, theta):
    """
    Implements Supervised PCA:
    1. Compute univariate regression coefficients for each feature with y.
    2. Reduce X to features whose coefficients exceed theta in absolute value.
    3. Compute first PC of reduced X.
    4. Return the first PC.
    """
    # 1. Compute univariate coefficients (standardized)
    # We'll use correlation which is equivalent to standardized regression coefficient for univariate
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0 for i in range(X.shape[1])])
    
    print("Num correlations < -theta: %d, > theta: %d" % (np.sum(correlations < -theta), np.sum(correlations > theta)) )

    # 2. Threshold features
    mask = np.abs(correlations) > theta
    if not np.any(mask):
        return None, mask
    
    X_reduced = X[:, mask]
    
    # 3. Compute first PC
    scaler = StandardScaler()
    X_reduced_scaled = scaler.fit_transform(X_reduced)
    
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_reduced_scaled)
    
    return pc1, mask

def main():
    print("Initializing Supervised PCA for Difficulty Prediction...")
    
    # 1. Generate Non-Whitened Vectors (Tag Proportions)
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_spca.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors_spca.npy"
    temp_constants_file = "research/temp_constants_spca.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Generating tag vectors...")
    try:
        vectors, appids = gtv.generate_tag_vectors(
            csv_path, 
            output_vectors=temp_vectors_file, 
            output_constants=temp_constants_file
        )
    except Exception as e:
        print(f"Error generating vectors: {e}")
        return

    # 2. Get Metadata
    print("Loading metadata...")
    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # 3. Extract "Difficult" and "Unforgiving" Tag Values for Target (y)
    target_tags = ["Difficult", "Unforgiving"]
    indices = []
    for tag in target_tags:
        if tag not in tag_to_idx:
            print(f"Error: '{tag}' tag not found.")
            return
        indices.append(tag_to_idx[tag])
    
    # Calculate y = z_diff + z_unforgiving
    # Using z-score transformation for blending
    y_raw = vectors[:, indices]
    y_z = (y_raw - np.mean(y_raw, axis=0)) / np.std(y_raw, axis=0)
    y = np.sum(y_z, axis=1)
    
    # 4. Prepare Features (X) - All tags except targets
    feature_mask = np.ones(vectors.shape[1], dtype=bool)
    for idx in indices:
        feature_mask[idx] = False
    X = vectors[:, feature_mask]
    feature_names = np.array(unique_tags)[feature_mask]
    
    print(f"Target: Blended Z-Score of {target_tags}")
    print(f"Number of features: {X.shape[1]}")
    
    # 5. Cross-Validation to find optimal theta
    print("Performing Cross-Validation for θ threshold...")
    # Possible thetas: based on correlation values
    # Let's check the range of correlations first
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0 for i in range(X.shape[1])])
    max_corr = np.max(np.abs(correlations))
    thetas = np.linspace(0, max_corr * 0.9, 20)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    theta_scores = []
    
    for theta in thetas:
        fold_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            pc1_train, mask = supervised_pca(X_train, y_train, theta)
            
            if pc1_train is None:
                fold_scores.append(-np.inf)
                continue
            
            # Train regression on PC1
            reg = LinearRegression()
            reg.fit(pc1_train, y_train)
            
            # Project test data using the same mask and a new PCA fit on test? 
            # No, supervised PCA usually projects using the loadings from the train set.
            X_test_reduced = X_test[:, mask]
            
            # We need the PCA object from the training step
            # Redo inside supervised_pca or refactor
            scaler = StandardScaler()
            X_train_reduced_scaled = scaler.fit_transform(X_train[:, mask])
            pca = PCA(n_components=1)
            pca.fit(X_train_reduced_scaled)
            
            X_test_reduced_scaled = scaler.transform(X_test_reduced)
            pc1_test = pca.transform(X_test_reduced_scaled)
            
            score = reg.score(pc1_test, y_test)
            fold_scores.append(score)
        
        theta_scores.append(np.mean(fold_scores))
        print(f"θ: {theta:.4f}, Mean R^2: {np.mean(fold_scores):.4f}")
        
    best_theta = thetas[np.argmax(theta_scores)]
    print(f"\nBest θ: {best_theta:.4f} with R^2: {np.max(theta_scores):.4f}")
    
    # 6. Final Model with Best Theta
    pc1, mask = supervised_pca(X, y, best_theta)
    selected_features = feature_names[mask]
    print(f"Selected {len(selected_features)} features.")
    if len(selected_features) > 0:
        print(f"First 10 selected features: {selected_features[:10]}")
    
    reg = LinearRegression()
    reg.fit(pc1, y)
    y_pred = reg.predict(pc1)
    
    # 7. Results and Predictors
    # In Supervised PCA, the "predictors" are the features with high loadings in the first PC
    # Get loadings
    scaler = StandardScaler()
    X_reduced_scaled = scaler.fit_transform(X[:, mask])
    pca = PCA(n_components=1)
    pca.fit(X_reduced_scaled)
    loadings = pca.components_[0]
    
    # The direction of the loadings depends on the sign of the regression coefficient
    if reg.coef_[0] < 0:
        loadings = -loadings
        
    loadings_df = pd.DataFrame({
        'feature': selected_features,
        'loading': loadings
    }).sort_values('loading', ascending=False)
    
    print("\nTop 20 Positive Predictors in PC1 (Harder):")
    print(loadings_df.head(20).to_string(index=False))
    
    print("\nTop 20 Negative Predictors in PC1 (Easier):")
    print(loadings_df.tail(20).to_string(index=False))
    
    # Predictions
    results = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, f"Unknown ({aid})") for aid in appids],
        'actual': y,
        'predicted': y_pred
    })
    
    print("\nTop 20 Predicted Hardest Games:")
    print(results.sort_values('predicted', ascending=False).head(20)[['name', 'predicted']].to_string(index=False))
    
    print("\nTop 20 Predicted Easiest Games:")
    print(results.sort_values('predicted', ascending=True).head(20)[['name', 'predicted']].to_string(index=False))
    
    # Save coefficients/loadings
    loadings_df.to_csv("research/supervised_pca_loadings_blended.csv", index=False)
    results.to_csv("research/supervised_pca_predictions_blended.csv", index=False)

    # Cleanup
    try:
        if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
        if os.path.exists(temp_constants_file): os.remove(temp_constants_file)
        if os.path.exists(gtv.W_TAG_FILE): os.remove(gtv.W_TAG_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
