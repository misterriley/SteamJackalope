
"""
Visualize the predicted probability of positive sentiment curve for a game.

This script uses the globally optimized parameters (gamma=0.5109, s=0.7812) to
generate a smooth prediction curve over a wide range of hypothetical playtimes.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import GLOBAL_POSITIVE_RATE

# Optimized global parameters
GLOBAL_GAMMA = 0.510897
GLOBAL_S = 0.781171

def predict_for_hypothetical_playtimes(train_playtimes, train_voted_up, test_playtimes, gamma, s, a=0.80):
    """
    Predict PPP for a set of hypothetical playtimes based on training data.
    
    Parameters:
    - train_playtimes: array of actual playtimes
    - train_voted_up: boolean array of actual votes
    - test_playtimes: array of hypothetical playtimes to predict for
    - gamma, s, a: model parameters
    
    Returns:
    - ppp: array of predicted probabilities for test_playtimes
    """
    # Precompute log playtimes
    eps = 1e-9
    log_train = np.log(train_playtimes + eps)
    log_test = np.log(test_playtimes + eps)
    
    # Compute weight matrix: (n_test, n_train)
    # Each row i contains weights for test point i against all training points j
    log_diff = log_test[:, np.newaxis] - log_train[np.newaxis, :]
    weight_matrix = np.exp(-0.5 * (log_diff / gamma) ** 2)
    
    # Sum weights (denominator)
    sum_all_weights = np.sum(weight_matrix, axis=1)
    
    # Sum positive weights (numerator)
    sum_pos_weights = np.sum(weight_matrix[:, train_voted_up], axis=1)
    
    # Bayesian smoothing
    numerator = sum_pos_weights + s * a
    denominator = sum_all_weights + s
    
    ppp = numerator / denominator
    
    return ppp

def main():
    # Load data
    reviews_path = 'scraped_reviews.csv'
    if not os.path.exists(reviews_path):
        print(f"Error: {reviews_path} not found.")
        return
        
    print(f"Loading reviews from {reviews_path}...")
    reviews_df = pd.read_csv(reviews_path)
    
    # Filter to valid reviews
    valid_reviews = reviews_df[
        (reviews_df['author_playtime_forever'] > 0) &
        (reviews_df['voted_up'].isin([True, False]))
    ]
    
    # Find top 10 games with most reviews
    game_counts = valid_reviews.groupby('appid').size().reset_index(name='count')
    top_10_games = game_counts.sort_values('count', ascending=False).head(10)
    
    print(f"Generating charts for top 10 games:")
    print(top_10_games.to_string(index=False))
    
    for _, row in top_10_games.iterrows():
        game_id = int(row['appid'])
        n_reviews = row['count']
        
        print(f"\nProcessing game: {game_id} ({n_reviews} reviews)...")
        
        # Extract training data
        game_reviews = valid_reviews[valid_reviews['appid'] == game_id]
        train_playtimes = game_reviews['author_playtime_forever'].values.astype(float)
        train_voted_up = game_reviews['voted_up'].values.astype(bool)
        
        # Determine range for hypothetical playtimes
        min_play = train_playtimes.min()
        max_play = train_playtimes.max()
        
        # Extend range slightly
        start_play = max(1.0, min_play * 0.5)
        end_play = max_play * 1.5
        
        # Create logarithmic grid
        grid_points = 1000
        hypothetical_playtimes = np.logspace(np.log10(start_play), np.log10(end_play), grid_points)
        
        # Predict
        ppp = predict_for_hypothetical_playtimes(
            train_playtimes, 
            train_voted_up, 
            hypothetical_playtimes, 
            gamma=GLOBAL_GAMMA, 
            s=GLOBAL_S, 
            a=GLOBAL_POSITIVE_RATE
        )
        
        # Visualization
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        # Plot the curve
        plt.plot(hypothetical_playtimes, ppp, color='blue', linewidth=3, label='Predicted Probability')
        
        # Plot actual reviews sample
        sample_size = min(len(train_playtimes), 1000)
        idx = np.random.choice(len(train_playtimes), sample_size, replace=False)
        scatter_play = train_playtimes[idx]
        scatter_vote = train_voted_up[idx]
            
        jitter = np.random.uniform(-0.05, 0.05, size=len(scatter_play))
        plt.scatter(scatter_play, scatter_vote.astype(float) + jitter, 
                    c=scatter_vote, cmap='coolwarm', alpha=0.2, s=8, label='Actual Reviews (Sample)')
        
        plt.xscale('log')
        plt.xlabel('Playtime (minutes) - Log Scale', fontsize=12)
        plt.ylabel('Probability of Positive Review', fontsize=12)
        plt.title(f'Sentiment Prediction Curve for Game {game_id}\n(Gamma={GLOBAL_GAMMA:.4f}, s={GLOBAL_S:.4f})', fontsize=14)
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        plt.ylim(-0.1, 1.1)
        plt.legend(loc='center right')
        
        # Add hours ticks
        hours = [0.1, 1, 10, 100, 1000, 10000]
        minutes = [h * 60 for h in hours]
        plt.xticks(minutes, [f'{h}h' for h in hours])
        
        output_file = f'research/playtime_curve_{game_id}.png'
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved to {output_file}")

if __name__ == "__main__":
    main()
