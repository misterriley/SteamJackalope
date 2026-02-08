import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pipeline.generate_tag_vectors as gtv

def main():
    print("Calculating Model Contributions per Game...")
    
    # 1. Load Coefficients and Selected Features
    coef_df = pd.read_csv("research/bic_stepwise_coefficients.csv")
    selected_features = coef_df['feature'].tolist()
    weights = coef_df['coefficient'].values
    
    # 2. Load Data and Metadata
    csv_path = "data/pipeline_games_clean.csv"
    df = gtv.load_data(csv_path)
    id_to_name = dict(zip(df['appid'], df['name']))
    _, tag_to_idx, unique_tags, _ = gtv.parse_tags(df)
    
    # 3. Reconstruct Vectors (Matching the processing in bic_difficulty_parallel)
    gtv.USE_TAG_WHITENING = False
    gtv.W_TAG_FILE = "research/temp_w_tag_contrib.npy"
    vectors, appids = gtv.generate_tag_vectors(csv_path, output_vectors="research/temp_vectors_contrib.npy")
    
    # Target and removal logic for zero-sum maintenance
    target_tags = ["Difficult", "Unforgiving"]
    target_indices = [tag_to_idx[t] for t in target_tags]
    removed_sums = np.sum(vectors[:, target_indices], axis=1)
    
    feature_mask = np.ones(vectors.shape[1], dtype=bool)
    for idx in target_indices: feature_mask[idx] = False
    
    X_full = vectors[:, feature_mask].copy()
    n_features_full = X_full.shape[1]
    
    # Zero-sum adjustment
    X_full = X_full + (removed_sums / n_features_full).reshape(-1, 1)
    
    # Subset to the exact columns used by the BIC model
    full_feature_names = np.array(unique_tags)[feature_mask]
    feature_to_idx_in_X = {name: i for i, name in enumerate(full_feature_names)}
    selected_indices_in_X = [feature_to_idx_in_X[name] for name in selected_features]
    
    X = X_full[:, selected_indices_in_X]
    
    # 4. Calculate Contributions
    # contribution = weight_i * value_i
    contributions = X * weights # Broadcasting weights across rows
    
    # 5. Create Result DataFrames
    contrib_df = pd.DataFrame(contributions, columns=selected_features)
    
    # Add metadata
    meta_df = pd.DataFrame({
        'appid': appids,
        'name': [id_to_name.get(aid, aid) for aid in appids]
    })
    
    final_df = pd.concat([meta_df, contrib_df], axis=1)
    
    # Calculate Total Predicted (should match previous results minus intercept)
    # We should add the intercept if we want full prediction
    # Get intercept from fitting the model one last time on the reconstructed X
    target_indices_raw = [tag_to_idx[t] for t in target_tags]
    y_raw = vectors[:, target_indices_raw]
    y_z = (y_raw - np.mean(y_raw, axis=0)) / np.std(y_raw, axis=0)
    y = np.sum(y_z, axis=1)
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    final_df['intercept'] = reg.intercept_
    final_df['total_predicted'] = final_df[selected_features].sum(axis=1) + final_df['intercept']
    final_df['actual_y'] = y
    
    # 6. Save and Display
    output_path = "research/model_contributions.parquet" # Use Parquet for high-dim data efficiency
    final_df.to_parquet(output_path, index=False)
    print(f"Saved detailed contributions to {output_path}")
    
    # Show examples for a high difficulty game
    hardest_game = final_df.sort_values('total_predicted', ascending=False).iloc[0]
    print(f"\nTop Contributor breakdown for: {hardest_game['name']}")
    
    # Get top 10 contributing tags for this game
    game_contribs = hardest_game[selected_features].astype(float).sort_values(ascending=False)
    print(game_contribs.head(10))
    print(f"Intercept: {hardest_game['intercept']:.4f}")
    print(f"Total Predicted: {hardest_game['total_predicted']:.4f}")
    
    # Cleanup
    for f in ["research/temp_w_tag_contrib.npy", "research/temp_vectors_contrib.npy"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
