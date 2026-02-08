import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def main():
    print("Initializing Refined Structural Ridge Difficulty Predictor...")
    
    # 1. Generate Non-Whitened Vectors
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_ridge.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors_ridge.npy"
    temp_constants_file = "research/temp_constants_ridge.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Generating vectors...")
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
    
    # 3. Prepare Target and Adjusted Features
    target_tag = "Difficult"
    if target_tag not in tag_to_idx:
        print(f"Error: '{target_tag}' tag not found.")
        return
    
    target_idx = tag_to_idx[target_tag]
    y_raw = vectors[:, target_idx]
    
    # Target: Difficult - mean(Difficult)
    y_mean = np.mean(y_raw)
    y = y_raw - y_mean
    
    # Features: Other tags
    X_indices = [i for i in range(len(unique_tags)) if i != target_idx]
    X_raw = vectors[:, X_indices]
    feature_names = [unique_tags[i] for i in X_indices]
    
    # Step A: Adjust features to remove structural 'Difficult' dependency
    num_features = X_raw.shape[1]
    print(f"Adjusting {num_features} features to remove 'Difficult' dependency...")
    X_adjustment = (y_raw / num_features).reshape(-1, 1)
    X_adj = X_raw + X_adjustment
    
    # Step B: Center features across games: X - mean(X)
    print("Centering features across games...")
    X_col_means = np.mean(X_adj, axis=0)
    X = X_adj - X_col_means
    
    # Verify properties
    row_sums = np.sum(X, axis=1)
    print(f"Max absolute row sum (target 0): {np.max(np.abs(row_sums)):.2e}")
    col_means = np.mean(X, axis=0)
    print(f"Max absolute column mean (target 0): {np.max(np.abs(col_means)):.2e}")
    
    # 4. Ridge CV with GridSearchCV for better control
    print("Running Ridge with 10-fold CV repeated 10 times (100 fits)...")
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    alphas = np.logspace(-3, 5, 50)
    
    ridge = Ridge(fit_intercept=False) # Intercept is 0
    param_grid = {'alpha': alphas}
    
    grid = GridSearchCV(ridge, param_grid, cv=cv, n_jobs=-1, scoring='r2')
    
    print("Fitting model...")
    grid.fit(X, y)
    
    model = grid.best_estimator_
    best_alpha = grid.best_params_['alpha']
    
    print(f"Best Alpha: {best_alpha:.6e}")
    print(f"R^2 Score on Train: {grid.best_score_:.4f}")
    
    # 5. Predict and Report
    print("Predicting difficulty...")
    y_pred_centered = model.predict(X)
    y_pred = y_pred_centered + y_mean
    
    results_list = []
    for i, appid in enumerate(appids):
        name = id_to_name.get(appid, f"Unknown ({appid})")
        results_list.append({
            "appid": appid,
            "name": name,
            "predicted_difficulty": y_pred[i],
            "actual_difficulty": y_raw[i]
        })
        
    results_df = pd.DataFrame(results_list)
    results_df.sort_values("predicted_difficulty", ascending=False, inplace=True)
    
    print("\nTop 20 Predicted Hardest Games:")
    print(results_df.head(20)[['name', 'predicted_difficulty', 'actual_difficulty']].to_string(index=False))
    
    print("\nTop 20 Predicted Easiest Games:")
    print(results_df.tail(20)[['name', 'predicted_difficulty', 'actual_difficulty']].to_string(index=False))

    coefs = model.coef_
    coef_results = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)
    
    print(f"\nSum of Coefficients (beta): {np.sum(coefs):.2e}")
    print(f"Mean of Coefficients: {np.mean(coefs):.2e}")
    
    print("\nTop 20 Positive Predictors:")
    for tag, val in coef_results[:20]:
        print(f"{tag}: {val:.6f}")

    print("\nTop 20 Negative Predictors:")
    for tag, val in coef_results[-20:]:
        print(f"{tag}: {val:.6f}")

    # Save results
    results_df.to_csv("research/ridge_difficulty_predictions.csv", index=False)
    coef_df = pd.DataFrame(coef_results, columns=['Tag', 'Coefficient'])
    coef_df.to_csv("research/ridge_difficulty_coefficients.csv", index=False)
    print("\nSaved predictions and coefficients to CSV files.")

    # Cleanup
    try:
        if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
        if os.path.exists(temp_constants_file): os.remove(temp_constants_file)
        if os.path.exists(gtv.W_TAG_FILE): os.remove(gtv.W_TAG_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
