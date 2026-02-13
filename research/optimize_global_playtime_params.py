"""
Optimize gamma and s globally across multiple games.

This script finds the kernel smoothing parameter (gamma) and regularization constant (s)
that maximize the total log-likelihood across a sample of games with many reviews.

For each game, we use a representative subset (default 200 reviews) to compute the
leave-one-out log-likelihood, and sum these across all selected games.

The optimized parameters can be used as defaults for the playtime sentiment model.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from itertools import product
from collections import defaultdict
import json

# Add parent directory to sys.path so we can import from analyze_playtime_sentiment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.analyze_playtime_sentiment import estimate_ppp_vectorized
from common.constants import GLOBAL_POSITIVE_RATE

def load_reviews_data():
    """Load the scraped reviews data."""
    possible_paths = [
        'scraped_reviews.csv',
        'data/scraped_reviews.csv',
        '../scraping/scraped_reviews.csv'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading reviews from {path}...")
            return pd.read_csv(path)
    
    raise FileNotFoundError("Could not find scraped_reviews.csv")

def get_games_with_min_reviews(reviews_df, min_reviews=200):
    """Return list of game IDs with at least min_reviews valid reviews."""
    valid_reviews = reviews_df[
        (reviews_df['author_playtime_forever'] > 0) &
        (reviews_df['voted_up'].isin([True, False]))
    ]
    
    game_counts = valid_reviews.groupby('appid').size().reset_index(name='count')
    eligible_games = game_counts[game_counts['count'] >= min_reviews]
    
    return eligible_games['appid'].tolist(), eligible_games

def compute_game_log_likelihood(playtimes, voted_up, gamma, s, a=0.80, sample_size=None):
    """
    Compute log-likelihood for a single game using leave-one-out predictions.
    
    Parameters:
    - playtimes: array of playtimes
    - voted_up: boolean array
    - gamma, s: model parameters
    - a: global positive rate (prior)
    - sample_size: if provided, randomly sample this many reviews (for speed)
    
    Returns:
    - log_likelihood: sum of log probabilities
    """
    n = len(playtimes)
    
    if sample_size is not None and n > sample_size:
        # Randomly sample without replacement
        idx = np.random.choice(n, size=sample_size, replace=False)
        playtimes_sub = playtimes[idx]
        voted_up_sub = voted_up[idx]
    else:
        playtimes_sub = playtimes
        voted_up_sub = voted_up
    
    # Compute PPP for all (sampled) reviews
    ppp = estimate_ppp_vectorized(playtimes_sub, voted_up_sub, gamma=gamma, s=s, a=a)
    
    # Compute log-likelihood
    ll = np.sum(voted_up_sub * np.log(ppp + 1e-12) + (~voted_up_sub) * np.log(1 - ppp + 1e-12))
    
    return ll

def optimize_global_parameters(reviews_df, game_ids, 
                              sample_per_game=200,
                              gamma_grid=None, s_grid=None,
                              verbose=True):
    """
    Optimize gamma and s by maximizing sum of log-likelihoods across games.
    
    Parameters:
    - reviews_df: full reviews dataframe
    - game_ids: list of game IDs to include in optimization
    - sample_per_game: number of reviews to sample per game (for computational efficiency)
    - gamma_grid: array of gamma values to search (default: logspace -3 to 2, 25 points)
    - s_grid: array of s values to search (default: logspace -3 to 3, 25 points)
    - verbose: print progress
    
    Returns:
    - best_gamma, best_s, best_ll
    """
    if gamma_grid is None:
        gamma_grid = np.logspace(-3, 2, 25)  # 0.001 to 100
    if s_grid is None:
        s_grid = np.logspace(-3, 3, 25)  # 0.001 to 1000
    
    # Preprocess: get playtimes and labels for each game
    if verbose:
        print("Preparing data for each game...")
    
    game_data = {}
    for game_id in game_ids:
        game_reviews = reviews_df[reviews_df['appid'] == game_id].copy()
        game_reviews = game_reviews[
            (game_reviews['author_playtime_forever'] > 0) &
            (game_reviews['voted_up'].isin([True, False]))
        ]
        
        if len(game_reviews) < 5:
            if verbose:
                print(f"  Skipping game {game_id}: only {len(game_reviews)} valid reviews")
            continue
        
        playtimes = game_reviews['author_playtime_forever'].values.astype(float)
        voted_up = game_reviews['voted_up'].values.astype(bool)
        
        game_data[game_id] = (playtimes, voted_up, len(game_reviews))
    
    if verbose:
        print(f"Total games in optimization: {len(game_data)}")
        total_reviews = sum(n for _, _, n in game_data.values())
        print(f"Total reviews across all games: {total_reviews}")
        print("\nStarting grid search...")
    
    best_ll = -np.inf
    best_gamma = None
    best_s = None
    
    total_combinations = len(gamma_grid) * len(s_grid)
    completed = 0
    
    # Grid search
    for gamma, reg_s in product(gamma_grid, s_grid):
        total_ll = 0.0
        
        # Sum log-likelihoods across all games
        for game_id, (playtimes, voted_up, n_reviews) in game_data.items():
            # Use sample_per_game reviews (or all if fewer)
            ll = compute_game_log_likelihood(
                playtimes, voted_up, 
                gamma=gamma, s=reg_s, 
                a=GLOBAL_POSITIVE_RATE,
                sample_size=sample_per_game
            )
            total_ll += ll
        
        if total_ll > best_ll:
            best_ll = total_ll
            best_gamma = gamma
            best_s = reg_s
        
        completed += 1
        if verbose and completed % 25 == 0:
            print(f"  Progress: {completed}/{total_combinations} - best ll={best_ll:.2f} (gamma={best_gamma:.4f}, s={best_s:.4f})")
    
    if verbose:
        print(f"\nOptimization complete!")
        print(f"Best gamma: {best_gamma:.6f}")
        print(f"Best s: {best_s:.6f}")
        print(f"Total log-likelihood: {best_ll:.2f}")
    
    return best_gamma, best_s, best_ll

def run_single_optimization(seed, sample_per_game=200, n_games=100):
    """Run a single optimization with given random seed."""
    np.random.seed(seed)
    
    # Load data
    reviews_df = load_reviews_data()
    
    # Find eligible games (≥200 reviews)
    eligible_games, game_counts = get_games_with_min_reviews(reviews_df, min_reviews=200)
    
    if len(eligible_games) == 0:
        raise ValueError("No games with enough reviews found.")
    
    # Randomly sample games
    if len(eligible_games) > n_games:
        sampled_games = np.random.choice(eligible_games, size=n_games, replace=False).tolist()
    else:
        sampled_games = eligible_games
    
    # Run optimization
    best_gamma, best_s, best_ll = optimize_global_parameters(
        reviews_df, 
        sampled_games,
        sample_per_game=sample_per_game,
        verbose=False  # suppress per-run output
    )
    
    return {
        'seed': seed,
        'gamma': float(best_gamma),
        's': float(best_s),
        'log_likelihood': float(best_ll),
        'n_games': len(sampled_games)
    }

def main():
    """Main entry point with multi-run support."""
    parser = argparse.ArgumentParser(description='Optimize playtime sentiment parameters globally across multiple games.')
    parser.add_argument('--n-runs', type=int, default=1,
                       help='Number of optimization runs with different random seeds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Starting random seed (for first run)')
    parser.add_argument('--sample-per-game', type=int, default=500,
                       help='Number of reviews to sample per game')
    parser.add_argument('--n-games', type=int, default=100,
                       help='Number of games to sample per run')
    parser.add_argument('--output', type=str, default='research/global_playtime_params_multi.json',
                       help='Output JSON file for all results')
    
    args = parser.parse_args()
    
    print(f"Running {args.n_runs} optimization runs...")
    print(f"Configuration: sample_per_game={args.sample_per_game}, n_games={args.n_games}")
    
    all_results = []
    
    for i in range(args.n_runs):
        current_seed = args.seed + i
        print(f"\n--- Run {i+1}/{args.n_runs} (seed={current_seed}) ---")
        
        result = run_single_optimization(
            seed=current_seed,
            sample_per_game=args.sample_per_game,
            n_games=args.n_games
        )
        
        all_results.append(result)
        print(f"Gamma: {result['gamma']:.6f}, s: {result['s']:.6f}, LL: {result['log_likelihood']:.2f}")
    
    # Collate statistics
    gammas = [r['gamma'] for r in all_results]
    ss = [r['s'] for r in all_results]
    lls = [r['log_likelihood'] for r in all_results]
    
    stats = {
        'gamma': {
            'mean': float(np.mean(gammas)),
            'std': float(np.std(gammas)),
            'min': float(np.min(gammas)),
            'max': float(np.max(gammas)),
            'values': gammas
        },
        's': {
            'mean': float(np.mean(ss)),
            'std': float(np.std(ss)),
            'min': float(np.min(ss)),
            'max': float(np.max(ss)),
            'values': ss
        },
        'log_likelihood': {
            'mean': float(np.mean(lls)),
            'std': float(np.std(lls)),
            'min': float(np.min(lls)),
            'max': float(np.max(lls)),
            'values': lls
        }
    }
    
    # Save full results
    output_data = {
        'configuration': {
            'n_runs': args.n_runs,
            'starting_seed': args.seed,
            'sample_per_game': args.sample_per_game,
            'n_games_per_run': args.n_games
        },
        'statistics': stats,
        'runs': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n=== MULTI-RUN OPTIMIZATION COMPLETE ===")
    print(f"Completed {args.n_runs} runs")
    print(f"\nParameter Stability Summary:")
    print(f"  Gamma: mean={stats['gamma']['mean']:.6f}, std={stats['gamma']['std']:.6f} (range: {stats['gamma']['min']:.6f} - {stats['gamma']['max']:.6f})")
    print(f"  s:     mean={stats['s']['mean']:.6f}, std={stats['s']['std']:.6f} (range: {stats['s']['min']:.6f} - {stats['s']['max']:.6f})")
    print(f"\nFull results saved to: {args.output}")

if __name__ == "__main__":
    main()