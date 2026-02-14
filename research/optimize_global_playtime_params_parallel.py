
"""
Optimize gamma and s globally across multiple games (Parallelized).

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
import multiprocessing
from functools import partial

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
    # Compute PPP for all reviews
    ppp = estimate_ppp_vectorized(playtimes, voted_up, gamma=gamma, s=s, a=a)
    
    # Compute log-likelihood (per-review cross-entropy)
    # ll = np.sum(voted_up * np.log(ppp + 1e-12) + (~voted_up) * np.log(1 - ppp + 1e-12))
    # We use np.clip to avoid log(0)
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

def run_single_optimization(seed, sample_per_game=200, n_games=None, grid_size=75):
    """Run a single optimization with given random seed."""
    if seed is not None:
        np.random.seed(seed)
    
    # Load data
    reviews_df = load_reviews_data()
    
    # Find eligible games (≥2 reviews)
    eligible_games, game_counts = get_games_with_min_reviews(reviews_df, min_reviews=2)
    
    if len(eligible_games) == 0:
        raise ValueError("No games with enough reviews found.")
    
    # Randomly sample games
    if n_games is not None and len(eligible_games) > n_games:
        sampled_games = np.random.choice(eligible_games, size=n_games, replace=False).tolist()
    else:
        sampled_games = eligible_games
    
    print(f"Preparing data for {len(sampled_games)} games...")
    
    # Prepare data for all games
    games_data = []
    # Optimization: Filter df once for all sampled appids
    valid_reviews = reviews_df[
        (reviews_df['appid'].isin(sampled_games)) &
        (reviews_df['author_playtime_forever'] > 0) &
        (reviews_df['voted_up'].isin([True, False]))
    ]
    
    for game_id, group in valid_reviews.groupby('appid'):
        playtimes = group['author_playtime_forever'].values.astype(float)
        voted_up = group['voted_up'].values.astype(bool)
        
        # Subsample if needed
        if sample_per_game is not None and len(playtimes) > sample_per_game:
            idx = np.random.choice(len(playtimes), size=sample_per_game, replace=False)
            playtimes = playtimes[idx]
            voted_up = voted_up[idx]
            
        games_data.append((playtimes, voted_up))
    
    # Define grid
    gammas = np.logspace(-3, 2, grid_size)  # 0.001 to 100
    s_values = np.logspace(-3, 3, grid_size)  # 0.001 to 1000
    param_grid = list(product(gammas, s_values))
    
    print(f"  Evaluating {len(param_grid)} parameter combinations on {len(games_data)} games...")
    
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    
    import time
    start_eval = time.time()
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Create partial function with fixed data
        func = partial(evaluate_parameters_on_games, games_data=games_data)
        
        # Map parameters to log-likelihoods
        results = pool.map(func, param_grid)
    
    end_eval = time.time()
    print(f"  Evaluation took {end_eval - start_eval:.2f}s")
    
    # Find best parameters
    best_idx = np.argmax(results)
    best_gamma, best_s = param_grid[best_idx]
    best_ll = results[best_idx]
    
    return {
        'seed': seed,
        'gamma': float(best_gamma),
        's': float(best_s),
        'log_likelihood': float(best_ll),
        'n_games': len(games_data),
        'duration': end_eval - start_eval
    }

def main():
    """Main entry point with multi-run support."""
    # Windows multiprocessing support
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='Optimize playtime sentiment parameters globally across multiple games.')
    parser.add_argument('--n-runs', type=int, default=1,
                       help='Number of optimization runs with different random seeds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Starting random seed (for first run)')
    parser.add_argument('--sample-per-game', type=int, default=200,
                       help='Number of reviews to sample per game')
    parser.add_argument('--n-games', type=int, default=None,
                       help='Number of games to sample per run (default: all)')
    parser.add_argument('--grid-size', type=int, default=75,
                       help='Number of steps in the gamma and s grid (total combinations = grid_size^2)')
    parser.add_argument('--output', type=str, default='research/global_playtime_params_multi.json',
                       help='Output JSON file for all results')
    
    args = parser.parse_args()
    
    print(f"Running {args.n_runs} optimization runs using multiprocessing...")
    print(f"Configuration: sample_per_game={args.sample_per_game}, n_games={args.n_games}")
    
    all_results = []
    
    for i in range(args.n_runs):
        current_seed = args.seed + i
        print(f"\n--- Run {i+1}/{args.n_runs} (seed={current_seed}) ---")
        
        result = run_single_optimization(
            seed=current_seed,
            sample_per_game=args.sample_per_game,
            n_games=args.n_games,
            grid_size=args.grid_size
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
