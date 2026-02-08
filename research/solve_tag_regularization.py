import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import ast
import os
from collections import Counter

def solve_tag_regularization(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Parsing tags...")
    all_game_tags = []
    for tag_str in df['tags']:
        if pd.isna(tag_str) or tag_str == '[]':
            all_game_tags.append({})
            continue
        try:
            # The tags are in a dict-like string format
            tags_dict = ast.literal_eval(tag_str)
            all_game_tags.append(tags_dict)
        except:
            all_game_tags.append({})
            
    # 1. Calculate Global Tag Distribution G
    print("Calculating global tag distribution...")
    global_counts = Counter()
    for tags in all_game_tags:
        global_counts.update(tags)
    
    total_global_votes = sum(global_counts.values())
    unique_tags = sorted(global_counts.keys())
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    num_tags = len(unique_tags)
    
    # G is the proportion of each tag across the whole dataset
    G = np.array([global_counts[tag] for tag in unique_tags], dtype=float) / total_global_votes
    
    # Pre-process game data for efficiency
    # Only keep games with > 100 votes for cross-validation
    cv_games_indices = []
    game_tag_matrices = [] # List of sparse-ish vectors (we'll use dense here for simplicity if it fits)
    
    print("Preparing games for cross-validation...")
    for i, tags in enumerate(all_game_tags):
        if not tags:
            continue
        
        votes = np.array(list(tags.values()))
        total_votes = votes.sum()
        
        if total_votes > 100:
            # Create a vector for this game
            vec = np.zeros(num_tags)
            for tag, count in tags.items():
                vec[tag_to_idx[tag]] = count
            
            game_tag_matrices.append(vec)
            cv_games_indices.append(i)
            
    game_tag_matrices = np.array(game_tag_matrices)
    num_cv_games = len(game_tag_matrices)
    print(f"Number of games for cross-validation: {num_cv_games}")
    
    # 2. Implement Cross-Validation loop
    # For each game, hide 10% of votes. 
    # We'll pre-split the data to keep it consistent during optimization.
    np.random.seed(42)
    training_votes = []
    hidden_votes = []
    
    for vec in game_tag_matrices:
        total = vec.sum()
        # Randomly select 10% of votes to hide
        # To simplify, we'll take 10% of the counts from each tag (approximate)
        # or more accurately, multinomial split
        h = np.random.multinomial(int(total * 0.1), vec / total)
        t = vec - h
        training_votes.append(t)
        hidden_votes.append(h)
        
    training_votes = np.array(training_votes)
    hidden_votes = np.array(hidden_votes)
    training_totals = training_votes.sum(axis=1, keepdims=True)

    # 3. Define Objective Function
    def log_likelihood(K):
        if K < 0: return 1e12
        
        # Smoothed Model: P(tag|game) = (training_tags + K*G) / (training_total + K)
        # We only care about the log-likelihood of the hidden votes
        
        probs = (training_votes + K * G) / (training_totals + K)
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        
        # Sum of hidden_votes * log(probs)
        ll = np.sum(hidden_votes * np.log(probs))
        return -ll # Minimize negative log-likelihood

    # 4. Use scipy.optimize.minimize_scalar
    print("Finding optimal K...")
    res = minimize_scalar(log_likelihood, bounds=(0, 1000), method='bounded')
    
    if res.success:
        optimal_K = res.x
        print(f"\nOptimal K: {optimal_K:.4f}")
        
        # 5. Return optimal K and final smoothed tag matrix
        # Final matrix using all votes
        all_totals = game_tag_matrices.sum(axis=1, keepdims=True)
        final_smoothed_matrix = (game_tag_matrices + optimal_K * G) / (all_totals + optimal_K)
        
        # Show top 5 tags for first CV game as example
        first_game_name = df.iloc[cv_games_indices[0]]['name']
        print(f"\nExample: Final Smoothed Tags for '{first_game_name}':")
        top_tags_idx = np.argsort(final_smoothed_matrix[0])[::-1][:5]
        for idx in top_tags_idx:
            print(f"  {unique_tags[idx]}: {final_smoothed_matrix[0][idx]:.4%}")
            
        return optimal_K, final_smoothed_matrix
    else:
        print("Optimization failed.")
        return None

if __name__ == "__main__":
    solve_tag_regularization("data/games_march2025_cleaned.csv")
