import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv
from common import constants

def main():
    parser = argparse.ArgumentParser(description="Calculate or Predict Difficulty Values")
    parser.add_argument("--mode", choices=["raw", "regression"], default="regression", 
                        help="Mode: 'raw' to print raw values, 'regression' to predict difficulty.")
    args = parser.parse_args()

    print("Initializing Difficulty Calculator...")
    
    # 1. Generate Non-Whitened Vectors
    # We monkey-patch to ensure we get regularized but non-whitened vectors
    print("Monkey-patching configuration for non-whitened vectors...")
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag.npy" 
    
    csv_path = "data/pipeline_games_clean.csv"
    temp_vectors_file = "research/temp_vectors.npy"
    temp_constants_file = "research/temp_constants.json"
    
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
        # Ensure alignment with vectors (generate_tag_vectors sorts and filters df? No, it drops duplicates and resets index)
        # load_data does the same, so order should match.
        # But generate_tag_vectors calls load_data internally.
        # vectors, appids are returned. We should map appids to names from df.
        
        # Create map from appid to name
        id_to_name = dict(zip(df['appid'], df['name']))
        
        _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return
    
    # 3. Extract Difficult Tag Value
    target_tag = "Difficult"
    if target_tag not in tag_to_idx:
        print(f"Error: '{target_tag}' tag not found.")
        return
    
    target_idx = tag_to_idx[target_tag]
    print(f"Extracting '{target_tag}' tag values...")
    
    difficult_values = vectors[:, target_idx]
    
    if args.mode == "raw":
        # 5. Save Results
        results = []
        for i, appid in enumerate(appids):
            name = id_to_name.get(appid, f"Unknown ({appid})")
            results.append({
                "appid": appid,
                "name": name,
                "difficulty_value": difficult_values[i]
            })
            
        results_df = pd.DataFrame(results)
        output_path = "research/difficulty_ratings_raw.csv"
        results_df.sort_values("difficulty_value", ascending=False, inplace=True)
        results_df.to_csv(output_path, index=False)
        
        print(f"Saved difficulty ratings to {output_path}")
        
        # 6. Show Examples
        print("\nTop 20 Games with Highest 'Difficult' Tag Value:")
        print(results_df.head(20)[['name', 'difficulty_value']].to_string(index=False))
        
        print("\nTop 20 Games with Lowest 'Difficult' Tag Value:")
        print(results_df.tail(20)[['name', 'difficulty_value']].to_string(index=False))

    elif args.mode == "regression":
        print("\nPerforming Constrained Regression to Predict Difficulty (Sum of Betas = 0)...")
        
        # Prepare X (features) and y (target)
        # X should be all tags EXCEPT "Difficult"
        
        # Create mask for features
        feature_mask = np.ones(vectors.shape[1], dtype=bool)
        feature_mask[target_idx] = False
        
        X_raw = vectors[:, feature_mask]
        y = difficult_values
        feature_names = np.array(unique_tags)[feature_mask]
        
        # Constraint Implementation: Sum(Beta) = 0
        # Transformation: Z_i = X_i - X_ref (where X_ref is the last feature)
        # Regression: y = alpha * Z + c
        # Result: beta_i = alpha_i, beta_ref = -sum(alpha)
        
        # Use the last feature as reference
        X_ref = X_raw[:, -1].reshape(-1, 1)
        X_others = X_raw[:, :-1]
        
        Z = X_others - X_ref
        
        # Fit Regression
        reg = LinearRegression()
        reg.fit(Z, y)
        
        # Reconstruct Betas
        betas_others = reg.coef_
        beta_last = -np.sum(betas_others)
        
        coefs = np.append(betas_others, beta_last)
        intercept = reg.intercept_
        
        # Calculate Predictions and R^2
        y_pred = X_raw @ coefs + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"\nR^2 Score: {r2:.4f}")
        print(f"Intercept (c): {intercept:.4f}")
        print(f"Check Sum of Betas: {np.sum(coefs):.4e}")
        
        sorted_coef_indices = np.argsort(coefs)
        
        # Show top 20 positive and negative predictors
        print("\nTop 20 Positive Predictors (Harder):")
        for idx in sorted_coef_indices[-20:][::-1]:
            print(f"{feature_names[idx]}: {coefs[idx]:.4f}")
    
        print("\nTop 20 Negative Predictors (Easier):")
        for idx in sorted_coef_indices[:20]:
            print(f"{feature_names[idx]}: {coefs[idx]:.4f}")
        
        # Predictions
        sorted_pred_indices = np.argsort(y_pred)
        
        # Show top 20 predicted hardest and easiest games
        print("\nTop 20 Predicted Hardest Games:")
        for idx in sorted_pred_indices[-20:][::-1]:
            appid = appids[idx]
            name = id_to_name.get(appid, f"Unknown ({appid})")
            print(f"{name}: {y_pred[idx]:.4f}")
        print("\nTop 20 Predicted Easiest Games:")
        for idx in sorted_pred_indices[:20]:
            appid = appids[idx]
            name = id_to_name.get(appid, f"Unknown ({appid})")
            print(f"{name}: {y_pred[idx]:.4f}") 

    # Cleanup
    try:
        if os.path.exists(temp_vectors_file): os.remove(temp_vectors_file)
        if os.path.exists(temp_constants_file): os.remove(temp_constants_file)
        if os.path.exists(gtv.W_TAG_FILE): os.remove(gtv.W_TAG_FILE)
    except:
        pass

if __name__ == "__main__":
    main()
