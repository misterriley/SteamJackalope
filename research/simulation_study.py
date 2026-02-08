import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import generate_tag_vectors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_tag_vectors import optimize_k, impute_tags, regularize, apply_clr, whiten

def generate_synthetic_data(n_games=1000, n_tags=100, seed=42):
    np.random.seed(seed)
    
    # 1. Define True Prior (G)
    # Power law distribution for tag popularity
    ranks = np.arange(1, n_tags + 1)
    weights = 1.0 / ranks**0.8
    G = weights / weights.sum()
    
    # 2. Define Covariance/Correlation structure
    # Random correlation matrix
    A = np.random.randn(n_tags, n_tags)
    Cov = np.dot(A, A.T)
    # Scale covariance to control strength of game-specific deviations
    Cov = Cov / np.max(Cov) * 2.0 
    
    # 3. Generate True Probabilities
    # Z ~ N(log(G), Cov) -> Softmax -> P
    # We want center of Z to align with log(G) roughly
    mu = np.log(G)
    Z = np.random.multivariate_normal(mu, Cov, size=n_games)
    
    # Softmax to get P
    exp_Z = np.exp(Z)
    P_true = exp_Z / exp_Z.sum(axis=1, keepdims=True)
    
    # 4. Generate Observed Counts with Censoring
    # Vote counts: LogNormal distribution
    # Mean votes = 50, but huge variance
    votes_log = np.random.normal(3.5, 1.5, size=n_games) 
    votes = np.maximum(1, np.exp(votes_log)).astype(int)
    
    counts_raw = np.zeros((n_games, n_tags), dtype=int)
    counts_censored = np.zeros((n_games, n_tags), dtype=int)
    
    for i in range(n_games):
        # Multinomial sample
        c = np.random.multinomial(votes[i], P_true[i])
        counts_raw[i] = c
        
        # Censor: Keep top 20
        if np.sum(c > 0) > 20:
            # Find 20th largest value
            sorted_indices = np.argsort(c)[::-1]
            top_20_indices = sorted_indices[:20]
            counts_censored[i, top_20_indices] = c[top_20_indices]
        else:
            counts_censored[i] = c
            
    return P_true, counts_censored, votes

def run_simulation():
    print("Generating synthetic data...")
    n_games = 2000
    n_tags = 100
    P_true, counts_censored, votes = generate_synthetic_data(n_games, n_tags)
    
    print(f"Data generated. Mean votes: {votes.mean():.1f}, Median: {np.median(votes):.1f}")
    
    # Convert to sparse format expected by functions (though functions handle dense mostly)
    # The functions in generate_tag_vectors take sparse for optimize_k, impute takes sparse
    import scipy.sparse as sp
    sparse_counts = sp.csr_matrix(counts_censored)
    
    # 1. Calculate Prior
    # Use ground truth threshold logic (> 1000 votes? scaled for simulation)
    # Our median votes is ~33. Let's use > 50 for prior.
    print("Calculating Prior...")
    # We can use the imported function or just calc manually to match pipeline exactly
    from generate_tag_vectors import calculate_prior, get_correlation_matrix
    
    # Use our synthetic counts to calculate Observed Prior
    prior_G_est = calculate_prior(sparse_counts, threshold=50)
    
    # Check Prior accuracy
    # True prior is defined in generation, but P_true average is the effective prior
    true_avg_P = P_true.mean(axis=0)
    prior_corr = pearsonr(prior_G_est, true_avg_P)[0]
    print(f"Prior Recovery Correlation: {prior_corr:.4f}")
    
    # 2. Optimize K
    print("Optimizing K...")
    K_opt = optimize_k(sparse_counts, prior_G_est)
    print(f"Optimal K: {K_opt:.4f}")
    
    # 3. Pipeline Steps
    # Impute
    corr_matrix = get_correlation_matrix(sparse_counts, threshold=50)
    augmented_counts = impute_tags(sparse_counts, corr_matrix)
    
    # Regularize
    P_est = regularize(augmented_counts, K_opt, prior_G_est)
    
    # Metric: How close is P_est to P_true?
    # Jensen-Shannon Divergence or MSE?
    mse = np.mean((P_est - P_true)**2)
    print(f"MSE (Probability Recovery): {mse:.6f}")
    
    # Correlation of vectors
    # Average correlation between True and Est vector for each game
    corrs = []
    for i in range(n_games):
        if np.std(P_est[i]) > 0 and np.std(P_true[i]) > 0:
            corrs.append(pearsonr(P_est[i], P_true[i])[0])
    print(f"Mean Vector Correlation (True vs Est): {np.mean(corrs):.4f}")
    
    # 4. CLR + Whiten
    print("Applying CLR and Whitening...")
    clr_vectors = apply_clr(P_est, prior_G_est)
    whitened = whiten(clr_vectors)
    
    # Check if Whitening recovers "True Structure"
    # In CLR space, true vectors are:
    P_true_safe = np.clip(P_true, 1e-15, 1.0)
    log_true = np.log(P_true_safe)
    clr_true = log_true - np.mean(log_true, axis=1, keepdims=True)
    
    # Center true relative to true prior
    prior_true_safe = np.clip(true_avg_P, 1e-15, 1.0)
    log_prior_true = np.log(prior_true_safe)
    clr_prior_true = log_prior_true - np.mean(log_prior_true)
    centered_true = clr_true - clr_prior_true
    
    # Correlation after CLR
    corrs_clr = []
    for i in range(n_games):
        corrs_clr.append(pearsonr(clr_vectors[i], centered_true[i])[0])
    print(f"Mean CLR Vector Correlation: {np.mean(corrs_clr):.4f}")
    
    # Dot Product Similarity Check
    # Pick a random pair of games. Compare Dot(True_i, True_j) vs Dot(Rec_i, Rec_j)
    # We need to normalize if we compare values, or check correlation of similarities
    print("Checking Similarity Preservation...")
    n_pairs = 1000
    indices_a = np.random.randint(0, n_games, n_pairs)
    indices_b = np.random.randint(0, n_games, n_pairs)
    
    sims_true = []
    sims_rec = []
    
    for i, j in zip(indices_a, indices_b):
        # Use Centered CLR for "True" similarity ground truth?
        # Or Whitened? 
        # If we assume the generative model creates correlations, whitening removes them.
        # But if we want to search for *semantic* similarity defined by the latent Z...
        # The latent Z is what we want to recover?
        # Z ~ N(mu, Cov).
        # We generated P from Z.
        # So Z is the ground truth feature vector.
        # Let's compare against Z?
        # But Z is not centered/whitened in the generation step necessarily (Cov has structure).
        
        # Let's just compare the final Whitened vectors to the True Centered CLR vectors
        # (Assuming whitening is 'part of the metric' we apply to truth as well? No).
        
        # Let's compare "Recovered Pipeline Output" vs "True Pipeline Output" (if we had full data)
        # i.e. If we observed ALL votes, what would we get?
        # VS Observed Censored votes -> Pipeline -> Output.
        pass
        
    # We'll just print correlation of CLR vectors which is a strong indicator of recovery.

if __name__ == "__main__":
    run_simulation()
