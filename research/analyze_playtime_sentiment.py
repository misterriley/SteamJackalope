"""
Analyze whether playtime can predict sentiment (positive/negative reviews) using a kernel smoothing approach.

For a given game with many reviews, this script implements a leave-one-out prediction where each review's
probability of being positive is estimated based on the weighted votes of other similar reviews, where
similarity is determined by how close their playtimes are (using a lognormal kernel).

The model:
- For a held-out review with playtime t_i, compute weights for all other reviews j:
  w_ij = exp(-(log(t_j) - log(t_i))^2 / (2*gamma^2))
  where gamma is the smoothing constant (bandwidth in log-space)
- PPP_i = (sum_{j: positive} w_ij + s * a) / (sum_all_j w_ij + s)
  where s is the regularization constant and a is the global positive rate (default 0.80)

This is a standard kernel regression / smoothing approach for binary outcomes.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from itertools import product

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import PLAYTIME_REGULARIZATION_C as PLAYTIME_C, GLOBAL_POSITIVE_RATE

def lognormal_kernel_vectorized(t_i, t_j, gamma):
    """
    Compute weight matrix between playtimes using a lognormal kernel (vectorized).
    
    Parameters:
    - t_i: scalar or array of playtimes (in minutes)
    - t_j: array of playtimes (in minutes)
    - gamma: smoothing constant (bandwidth in log-space)
    
    Returns:
    - weights: similarity scores (scalar if t_i is scalar, or array if t_i is array)
    """
    eps = 1e-9
    log_t_i = np.log(np.atleast_1d(t_i) + eps)
    log_t_j = np.log(t_j + eps)
    
    # For scalar t_i, compute against all t_j
    if np.isscalar(t_i):
        z = (log_t_j - log_t_i) / gamma
        return np.exp(-0.5 * z * z)
    else:
        # For vector t_i, compute pairwise matrix
        # Shape: (len(t_i), len(t_j))
        log_t_i_col = log_t_i[:, np.newaxis]  # column vector
        log_t_j_row = log_t_j[np.newaxis, :]  # row vector
        z = (log_t_j_row - log_t_i_col) / gamma
        return np.exp(-0.5 * z * z)

def estimate_ppp_vectorized(playtimes, voted_up, gamma, s, a=0.80, leave_out_idx=None):
    """
    Estimate the Predicted Probability of Positive for reviews using vectorized operations.
    
    Parameters:
    - playtimes: array of playtimes in minutes
    - voted_up: boolean array (True=positive, False=negative)
    - gamma: smoothing constant (bandwidth)
    - s: regularization constant (like in Bayesian smoothing)
    - a: global positive rate (prior)
    - leave_out_idx: index of the review to predict, or None for all LOO
    
    Returns:
    - ppp: array of predicted probabilities
    """
    n = len(playtimes)
    
    # Precompute log playtimes for efficiency
    eps = 1e-9
    log_playtimes = np.log(playtimes + eps)
    
    # Compute full weight matrix (n x n) - off-diagonal entries are the relevant weights
    # w_ij = exp(-(log(t_j) - log(t_i))^2 / (2*gamma^2))
    log_diff = log_playtimes[:, np.newaxis] - log_playtimes[np.newaxis, :]  # shape (n, n)
    weight_matrix = np.exp(-0.5 * (log_diff / gamma) ** 2)
    
    # Zero out diagonal (self-weights)
    np.fill_diagonal(weight_matrix, 0.0)
    
    # Fully vectorized sum calculation (avoids Python loop)
    # Sum of weights for all reviews (denominator term)
    sum_all_weights = np.sum(weight_matrix, axis=1)
    
    # Sum of weights for positive reviews (numerator term)
    # Only sum columns where voted_up is True
    sum_pos_weights = np.sum(weight_matrix[:, voted_up], axis=1)
    
    # Apply Bayesian smoothing: add s*a to numerator, s to denominator
    numerator = sum_pos_weights + s * a
    denominator = sum_all_weights + s
    
    # Handle zero denominator case (fallback to prior)
    denominator = np.maximum(denominator, 1e-12)
    
    ppp = numerator / denominator
    
    if leave_out_idx is not None:
        return ppp[[leave_out_idx]]
    
    return ppp

def estimate_ppp(playtimes, voted_up, gamma, s, a=0.80, leave_out_idx=None):
    """
    Estimate the Predicted Probability of Positive for a target review.
    Wrapper around vectorized implementation.
    
    If leave_out_idx is None, calculates PPP for all reviews using leave-one-out.
    If leave_out_idx is an integer, calculates PPP for that specific review
    using all other reviews as the training set.
    
    Parameters:
    - playtimes: array of playtimes in minutes
    - voted_up: boolean array (True=positive, False=negative)
    - gamma: smoothing constant (bandwidth)
    - s: regularization constant (like in Bayesian smoothing)
    - a: global positive rate (prior)
    - leave_out_idx: index of the review to predict, or None for all LOO
    
    Returns:
    - ppp: array of predicted probabilities
    """
    return estimate_ppp_vectorized(playtimes, voted_up, gamma, s, a, leave_out_idx)

def analyze_game(game_id, gamma=None, s=None, reviews_df=None, output_dir='research/playtime_analysis', 
                 use_full_dataset=True, n_folds=10, verbose=True):
    """
    Analyze a single game to determine if playtime predicts sentiment.
    
    Parameters:
    - game_id: Steam appid
    - gamma: smoothing constant (bandwidth in log-space). If None, will be optimized.
    - s: regularization constant. If None, uses PLAYTIME_REGULARIZATION_C or 100.0
    - reviews_df: DataFrame with review data (if None, loads from default path)
    - output_dir: directory to save plots and results
    - use_full_dataset: if True, use ALL reviews for optimization (slower but more accurate)
    - n_folds: number of folds for cross-validation (used if use_full_dataset is False)
    - verbose: print progress
    
    Returns:
    - results: dict with analysis metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reviews if not provided
    if reviews_df is None:
        possible_paths = [
            'scraped_reviews.csv',
            'data/scraped_reviews.csv',
            '../scraping/scraped_reviews.csv'
        ]
        reviews_path = None
        for path in possible_paths:
            if os.path.exists(path):
                reviews_path = path
                break
        
        if reviews_path is None:
            raise FileNotFoundError("Could not find scraped_reviews.csv")
        
        if verbose:
            print(f"Loading reviews from {reviews_path}...")
        reviews_df = pd.read_csv(reviews_path)
    
    # Filter to this game
    game_reviews = reviews_df[reviews_df['appid'] == game_id].copy()
    
    if len(game_reviews) < 10:
        if verbose:
            print(f"Warning: Only {len(game_reviews)} reviews found for game {game_id}. Results may be unreliable.")
    
    # Filter to positive playtime and valid sentiment
    game_reviews = game_reviews[
        (game_reviews['author_playtime_forever'] > 0) &
        (game_reviews['voted_up'].isin([True, False]))
    ].copy()
    
    if len(game_reviews) < 5:
        raise ValueError(f"Not enough valid reviews for game {game_id} after filtering.")
    
    # Convert to arrays
    playtimes = game_reviews['author_playtime_forever'].values.astype(float)
    voted_up = game_reviews['voted_up'].values.astype(bool)
    
    if verbose:
        print(f"Game ID: {game_id}")
        print(f"Number of reviews: {len(game_reviews)}")
        print(f"Positive: {voted_up.sum()}, Negative: {(~voted_up).sum()}")
        print(f"Playtime range: {playtimes.min():.1f} to {playtimes.max():.1f} minutes")
        print(f"Median playtime: {np.median(playtimes):.1f}")
    
    # Determine if we need to optimize parameters
    need_optimize = (gamma is None) or (s is None)
    
    if need_optimize:
        if verbose:
            print("\nOptimizing gamma and s...")
        
        # Define parameter grids
        gammas = np.logspace(-3, 2, 25)  # 0.001 to 100 in logspace
        s_values = np.logspace(-3, 3, 25)  # 0.001 to 1000 in logspace
        
        # Use full dataset if requested and not too large, otherwise use subset or CV
        n_reviews = len(playtimes)
        if use_full_dataset and n_reviews <= 500:
            # Use full leave-one-out on all reviews (computationally intensive but feasible)
            if verbose:
                print(f"Using full dataset ({n_reviews} reviews) for grid search...")
            best_ll = -np.inf
            best_gamma = None
            best_s = None
            
            for g, reg_s in product(gammas, s_values):
                ppp = estimate_ppp_vectorized(playtimes, voted_up, gamma=g, s=reg_s, a=GLOBAL_POSITIVE_RATE)
                ll = np.sum(voted_up * np.log(ppp + 1e-12) + (~voted_up) * np.log(1 - ppp + 1e-12))
                
                if ll > best_ll:
                    best_ll = ll
                    best_gamma = g
                    best_s = reg_s
        elif use_full_dataset and n_reviews > 500:
            # Too large, fall back to subset
            if verbose:
                print(f"Dataset too large ({n_reviews} reviews). Using subset of 200 for optimization...")
            max_opt = min(200, n_reviews)
            subset_idx = np.random.choice(n_reviews, size=max_opt, replace=False)
            playtimes_sub = playtimes[subset_idx]
            voted_up_sub = voted_up[subset_idx]
            
            best_ll = -np.inf
            best_gamma = None
            best_s = None
            
            for g, reg_s in product(gammas, s_values):
                ppp = estimate_ppp_vectorized(playtimes_sub, voted_up_sub, gamma=g, s=reg_s, a=GLOBAL_POSITIVE_RATE)
                ll = np.sum(voted_up_sub * np.log(ppp + 1e-12) + (~voted_up_sub) * np.log(1 - ppp + 1e-12))
                
                if ll > best_ll:
                    best_ll = ll
                    best_gamma = g
                    best_s = reg_s
        else:
            # Use k-fold cross-validation
            if verbose:
                print(f"Using {n_folds}-fold cross-validation for grid search...")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            best_ll = -np.inf
            best_gamma = None
            best_s = None
            
            for g, reg_s in product(gammas, s_values):
                fold_lls = []
                for train_idx, val_idx in kf.split(playtimes):
                    train_play = playtimes[train_idx]
                    train_voted = voted_up[train_idx]
                    val_play = playtimes[val_idx]
                    val_voted = voted_up[val_idx]
                    
                    # Compute PPP for validation set using train as reference
                    # For each val point, compute weights from all train points
                    eps = 1e-9
                    log_train = np.log(train_play + eps)
                    log_val = np.log(val_play + eps)
                    
                    # Compute weight matrix: val x train
                    log_diff = log_val[:, np.newaxis] - log_train[np.newaxis, :]
                    weights = np.exp(-0.5 * (log_diff / g) ** 2)
                    
                    # Compute PPP for each validation point
                    numerator = (weights * train_voted).sum(axis=1) + reg_s * a
                    denominator = weights.sum(axis=1) + reg_s
                    ppp_val = numerator / denominator
                    
                    ll = np.sum(val_voted * np.log(ppp_val + 1e-12) + (~val_voted) * np.log(1 - ppp_val + 1e-12))
                    fold_lls.append(ll)
                
                mean_ll = np.mean(fold_lls)
                if mean_ll > best_ll:
                    best_ll = mean_ll
                    best_gamma = g
                    best_s = reg_s
        
        if verbose:
            print(f"Selected gamma={best_gamma:.4f}, s={best_s:.4f} (max log-likelihood={best_ll:.2f})")
        
        # Set optimized values only if they were originally None
        if gamma is None:
            gamma = best_gamma
        if s is None:
            s = best_s
    
    # If after optimization either parameter is still None, assign defaults
    if s is None:
        s = PLAYTIME_C if PLAYTIME_C is not None else 100.0
    if gamma is None:
        gamma = 1.0
    
    # Calculate PPP for all reviews using LOO (vectorized for speed)
    if verbose:
        print(f"\nCalculating leave-one-out PPP predictions with gamma={gamma:.4f}, s={s:.4f}...")
    ppp = estimate_ppp_vectorized(playtimes, voted_up, gamma=gamma, s=s, a=GLOBAL_POSITIVE_RATE)
    
    # Add predictions to dataframe
    game_reviews = game_reviews.copy()
    game_reviews['ppp'] = ppp
    
    # Calculate classification metrics
    predicted_positive = ppp > 0.5
    actual_positive = voted_up
    
    accuracy = np.mean(predicted_positive == actual_positive)
    precision = np.sum((predicted_positive & actual_positive)) / (np.sum(predicted_positive) + 1e-12)
    recall = np.sum((predicted_positive & actual_positive)) / (np.sum(actual_positive) + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    if verbose:
        print("\n--- Classification Performance (threshold=0.5) ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Correlation between playtime and sentiment
        sentiment_numeric = voted_up.astype(float)
        correlation = np.corrcoef(np.log1p(playtimes), sentiment_numeric)[0, 1]
        print(f"\nCorrelation (log playtime vs positive sentiment): {correlation:.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Smoothed density of PPP colored by actual sentiment
    ax = axes[0, 0]
    sns.kdeplot(data=game_reviews, x='ppp', hue='voted_up', fill=True, common_norm=False, ax=ax)
    ax.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
    ax.set_xlabel('Predicted Probability of Positive (PPP)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of PPP by Actual Sentiment')
    ax.legend()
    
    # 2. Playtime distributions for positive vs negative reviews
    ax = axes[0, 1]
    pos_play = playtimes[voted_up]
    neg_play = playtimes[~voted_up]
    sns.kdeplot(x=pos_play, label='Positive', fill=True, ax=ax)
    if len(neg_play) > 0:
        sns.kdeplot(x=neg_play, label='Negative', fill=True, ax=ax)
    ax.set_xlabel('Playtime (minutes)')
    ax.set_ylabel('Density')
    ax.set_title('Playtime Distribution by Sentiment')
    ax.legend()
    ax.set_xscale('log')
    
    # 3. Scatter: playtime vs PPP colored by actual sentiment
    ax = axes[1, 0]
    scatter = ax.scatter(playtimes, ppp, c=voted_up, cmap='coolwarm', alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Playtime (minutes)')
    ax.set_ylabel('Predicted Probability Positive')
    ax.set_title('Playtime vs PPP by Actual Sentiment')
    ax.set_xscale('log')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Actual (1=Positive)')
    
    # 4. Binned PPP by playtime quantiles
    ax = axes[1, 1]
    n_bins = 10
    game_reviews['playtime_q'] = pd.qcut(game_reviews['author_playtime_forever'], q=n_bins, duplicates='drop')
    binned = game_reviews.groupby('playtime_q').agg({
        'ppp': 'mean',
        'voted_up': 'mean',
        'author_playtime_forever': 'mean'
    }).reset_index()
    
    ax.plot(binned['author_playtime_forever'], binned['ppp'], 'o-', label='Mean PPP', linewidth=2)
    ax.plot(binned['author_playtime_forever'], binned['voted_up'], 's-', label='Actual Positive Rate', linewidth=2)
    ax.set_xlabel('Playtime (minutes) - binned quantiles')
    ax.set_ylabel('Probability')
    ax.set_title('PPP vs Actual by Playtime Quantile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'playtime_sentiment_{game_id}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"\nPlot saved to {plot_path}")
    plt.close()
    
    # Save detailed results
    results = {
        'game_id': game_id,
        'n_reviews': len(game_reviews),
        'n_positive': int(voted_up.sum()),
        'n_negative': int((~voted_up).sum()),
        'gamma': float(gamma),
        's': float(s),
        'global_positive_rate': float(GLOBAL_POSITIVE_RATE),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'correlation_log_playtime_sentiment': float(correlation),
        'median_playtime': float(np.median(playtimes)),
        'output_dir': output_dir
    }
    
    # Save predictions CSV
    predictions_csv = os.path.join(output_dir, f'predictions_{game_id}.csv')
    game_reviews[['review_id', 'author_playtime_forever', 'voted_up', 'ppp']].to_csv(predictions_csv, index=False)
    if verbose:
        print(f"Predictions saved to {predictions_csv}")
    
    return results, game_reviews

def main():
    """
    Main entry point: pick a game with many reviews and analyze it.
    """
    # Try to load reviews
    possible_paths = [
        'scraped_reviews.csv',
        'data/scraped_reviews.csv',
        '../scraping/scraped_reviews.csv'
    ]
    reviews_path = None
    for path in possible_paths:
        if os.path.exists(path):
            reviews_path = path
            break
    
    if reviews_path is None:
        print("Error: Could not find scraped_reviews.csv")
        return
    
    print(f"Loading reviews from {reviews_path}...")
    reviews_df = pd.read_csv(reviews_path)
    
    # Ensure required columns exist
    required_cols = ['appid', 'author_playtime_forever', 'voted_up']
    missing = [c for c in required_cols if c not in reviews_df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return
    
    # Filter to reviews with valid playtime
    valid_reviews = reviews_df[
        (reviews_df['author_playtime_forever'] > 0) &
        (reviews_df['voted_up'].isin([True, False]))
    ]
    
    # Find games with many reviews
    game_counts = valid_reviews.groupby('appid').size().reset_index(name='count')
    game_counts = game_counts.sort_values('count', ascending=False)
    
    print("\nTop 10 games by number of valid reviews:")
    print(game_counts.head(10).to_string(index=False))
    
    # Pick the game with the most reviews
    top_game = game_counts.iloc[0]
    game_id = int(top_game['appid'])
    n_reviews = top_game['count']
    
    print(f"\nSelected game {game_id} with {n_reviews} valid reviews.")
    
    # Run analysis with full dataset for optimization
    results, detailed_df = analyze_game(
        game_id=game_id,
        gamma=None,  # auto-optimize
        s=None,  # auto-optimize jointly with gamma
        reviews_df=reviews_df,
        output_dir='research/playtime_analysis',
        use_full_dataset=True,  # Use all reviews for optimization
        n_folds=10,
        verbose=True
    )
    
    # Print summary
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Game ID: {game_id}")
    print(f"Total reviews used: {results['n_reviews']}")
    print(f"Optimal gamma: {results['gamma']:.4f}")
    print(f"Regularization s: {results['s']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Correlation (log playtime vs positive): {results['correlation_log_playtime_sentiment']:.4f}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if results['correlation_log_playtime_sentiment'] > 0.1:
        print("Positive correlation: longer playtime is associated with positive reviews.")
    elif results['correlation_log_playtime_sentiment'] < -0.1:
        print("Negative correlation: shorter playtime is associated with positive reviews (players quitting early).")
    else:
        print("Weak correlation: playtime alone is not strongly predictive of sentiment.")

if __name__ == "__main__":
    main()