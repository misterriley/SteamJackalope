import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def main():
    print("Initializing LASSO Difficulty Predictor...")
    
    # 1. Generate Non-Whitened Vectors
    print("Monkey-patching configuration for non-whitened vectors...")
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_lasso.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors_lasso.npy"
    temp_constants_file = "research/temp_constants_lasso.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Generating vectors (may take a minute)...")
    try:
        vectors, appids = gtv.generate_tag_vectors(
            csv_path, 
            output_vectors=temp_vectors_file, 
            output_constants=temp_constants_file
        )
    except Exception as e:
        print(f"Error generating vectors: {e}")
        return

    # 2. Get Metadata (Names and Tags)
    print("Loading metadata...")
    try:
        df = gtv.load_data(csv_path)
        # Create map from appid to name
        id_to_name = dict(zip(df['appid'], df['name']))
        _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return
    
    # 3. Setup Regression
    target_tag = "Difficult"
    if target_tag not in tag_to_idx:
        print(f"Error: '{target_tag}' tag not found.")
        return
    
    target_idx = tag_to_idx[target_tag]
    print(f"Target: '{target_tag}' (Index {target_idx})")
    
    y = vectors[:, target_idx]
    X_indices = [i for i in range(len(unique_tags)) if i != target_idx]
    X = vectors[:, X_indices]
    
    feature_names = [unique_tags[i] for i in X_indices]
    
    # 4. LASSO CV
    print("Running LASSO with 4-fold CV averaged over 6 runs...")
    cv = RepeatedKFold(n_splits=4, n_repeats=6, random_state=42)
    
    model = LassoCV(
        cv=cv, 
        n_jobs=-1, 
        max_iter=5000,
        tol=1e-3,
        verbose=1,
        selection='random'
    )
    
    print("Fitting model...")
    model.fit(X, y)
    
    print(f"Best Alpha: {model.alpha_:.6e}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R^2 Score: {model.score(X, y):.4f}")
    
    # 5. Predict for all games
    print("Predicting difficulty for all games...")
    y_pred = model.predict(X)
    
    # 6. Report results
    results_list = []
    for i, appid in enumerate(appids):
        name = id_to_name.get(appid, f"Unknown ({appid})")
        results_list.append({
            "appid": appid,
            "name": name,
            "predicted_difficulty": y_pred[i],
            "actual_difficulty": y[i]
        })
        
    results_df = pd.DataFrame(results_list)
    results_df.sort_values("predicted_difficulty", ascending=False, inplace=True)
    
    print("\nTop 20 Predicted Hardest Games:")
    print(results_df.head(20)[['name', 'predicted_difficulty', 'actual_difficulty']].to_string(index=False))
    
    print("\nTop 20 Predicted Easiest Games:")
    print(results_df.tail(20)[['name', 'predicted_difficulty', 'actual_difficulty']].to_string(index=False))

    # Optional: print top coefficients
    coefs = model.coef_
    coef_results = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Positive Predictors:")
    for tag, val in coef_results[:20]:
        print(f"{tag}: {val:.6f}")

    print("\nTop 20 Negative Predictors:")
    for tag, val in coef_results[-20:]:
        print(f"{tag}: {val:.6f}")

    # Save results to CSV for the user to see
    results_df.to_csv("research/lasso_difficulty_predictions.csv", index=False)
    print("\nSaved predictions to research/lasso_difficulty_predictions.csv")
    
    coef_df = pd.DataFrame(coef_results, columns=['Tag', 'Coefficient'])
    coef_df.to_csv("research/lasso_difficulty_coefficients.csv", index=False)
    print("Saved coefficients to research/lasso_difficulty_coefficients.csv")

    # Cleanup
    try:
        if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
        if os.path.exists(temp_constants_file): os.remove(temp_constants_file)
        if os.path.exists(gtv.W_TAG_FILE): os.remove(gtv.W_TAG_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
