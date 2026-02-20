"""
Generates and stores optimal global playtime sentiment parameters (gamma and s).

This script adapts the optimization logic from research/optimize_global_playtime_params_parallel.py
to be integrated into the main pipeline. It finds the kernel smoothing parameter (gamma)
and regularization constant (s) that maximize the total log-likelihood across a sample of games
with many reviews.

The optimal parameters are saved to pipeline/regularization_constants.json.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import multiprocessing
from functools import partial
from itertools import product
import time

# Add parent directory to sys.path so we can import from research and common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.analyze_playtime_sentiment import estimate_ppp_vectorized
from common.constants import GLOBAL_POSITIVE_RATE, ROOT_DIR, REGULARIZATION_FILE

def load_reviews_data():
    """Load the scraped reviews data."""
    possible_paths = [
        os.path.join(ROOT_DIR, 'scraped_reviews.csv'),
        os.path.join(ROOT_DIR, 'data', 'scraped_reviews.csv'),
        os.path.join(ROOT_DIR, 'scraping', 'scraped_reviews.csv')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading reviews from {path}...")
            return pd.read_csv(path)
    
    raise FileNotFoundError("Could not find scraped_reviews.csv")

def get_games_with_min_reviews(reviews_df, min_reviews=2):
    """Return list of game IDs with at least min_reviews valid reviews."""
    valid_reviews = reviews_df[
        (reviews_df['author_playtime_forever'] > 0) &
        (reviews_df['voted_up'].isin([True, False]))
    ]
    
    game_counts = valid_reviews.groupby('appid').size().reset_index(name='count')
    eligible_games = game_counts[game_counts['count'] >= min_reviews]
    
    return eligible_games['appid'].tolist(), eligible_games

def compute_game_log_likelihood(playtimes, voted_up, gamma, s, a=0.80):
    """
    Compute log-likelihood for a single game using leave-one-out predictions.
    Helper function for parallel execution.
    """
    ppp = estimate_ppp_vectorized(playtimes, voted_up, gamma=gamma, s=s, a=a)
    ppp_clipped = np.clip(ppp, 1e-12, 1.0 - 1e-12)
    ll = np.sum(voted_up * np.log(ppp_clipped) + (~voted_up) * np.log(1 - ppp_clipped))
    return ll

def evaluate_parameters_on_games(params_tuple, games_data):
    """
    Evaluate a single (gamma, s) pair across all games.
    Used for parallelization.
    """
    gamma, s = params_tuple
    total_ll = 0.0
    for playtimes, voted_up in games_data:
        ll = compute_game_log_likelihood(playtimes, voted_up, gamma, s, GLOBAL_POSITIVE_RATE)
        total_ll += ll
    return total_ll

def generate_playtime_params(seed=42, sample_per_game=200, n_games=100, grid_size=75, output_file=REGULARIZATION_FILE):
    """
    Runs the optimization for playtime sentiment parameters and saves them.
    """
    print("Starting playtime sentiment parameter optimization...")
    np.random.seed(seed)
    
    reviews_df = load_reviews_data()
    
    eligible_games, game_counts = get_games_with_min_reviews(reviews_df, min_reviews=2)
    
    if len(eligible_games) == 0:
        raise ValueError("No games with enough reviews found for playtime parameter optimization.")
    
    if n_games is not None and len(eligible_games) > n_games:
        sampled_games = np.random.choice(eligible_games, size=n_games, replace=False).tolist()
    else:
        sampled_games = eligible_games
    
    print(f"Preparing data for {len(sampled_games)} games for optimization...")
    
    games_data = []
    valid_reviews = reviews_df[
        (reviews_df['appid'].isin(sampled_games)) &
        (reviews_df['author_playtime_forever'] > 0) &
        (reviews_df['voted_up'].isin([True, False]))
    ]
    
    for game_id, group in valid_reviews.groupby('appid'):
        playtimes = group['author_playtime_forever'].values.astype(float)
        voted_up = group['voted_up'].values.astype(bool)
        
        if sample_per_game is not None and len(playtimes) > sample_per_game:
            idx = np.random.choice(len(playtimes), size=sample_per_game, replace=False)
            playtimes = playtimes[idx]
            voted_up = voted_up[idx]
            
        games_data.append((playtimes, voted_up))
    
    gammas = np.logspace(-3, 2, grid_size)  # 0.001 to 100
    s_values = np.logspace(-3, 3, grid_size)  # 0.001 to 1000
    param_grid = list(product(gammas, s_values))
    
    print(f"Evaluating {len(param_grid)} parameter combinations on {len(games_data)} games...")
    
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    
    start_eval = time.time()
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        func = partial(evaluate_parameters_on_games, games_data=games_data)
        results = pool.map(func, param_grid)
    
    end_eval = time.time()
    print(f"Evaluation took {end_eval - start_eval:.2f}s")
    
    best_idx = np.argmax(results)
    best_gamma, best_s = param_grid[best_idx]
    best_ll = results[best_idx]
    
    print(f"Optimal Gamma: {best_gamma:.6f}, Optimal s: {best_s:.6f}, Max Log-Likelihood: {best_ll:.2f}")

    # Load existing regularization constants or initialize empty
    reg_constants = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            reg_constants = json.load(f)
    
    # Update with new values
    reg_constants['PLAYTIME_SENTIMENT_GAMMA'] = float(best_gamma)
    reg_constants['PLAYTIME_SENTIMENT_S'] = float(best_s)
    
    # Save updated constants
    with open(output_file, 'w') as f:
        json.dump(reg_constants, f, indent=4)
        
    print(f"Updated playtime sentiment parameters in {output_file}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    generate_playtime_params()
