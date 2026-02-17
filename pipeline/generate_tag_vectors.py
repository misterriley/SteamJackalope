import pandas as pd
import numpy as np
import ast
import os
import json
from tqdm import tqdm
from collections import Counter
from scipy.optimize import minimize_scalar
import scipy.sparse as sp
from scipy.stats import chi
import sys

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    TAG_EM_ITERATIONS, 
    TAG_EM_RIDGE, 
    TAG_OPT_SAMPLE_SIZE,
    CHI_FIT_NORM_THRESHOLD,
    CHI_FIT_PERCENTILE,
    USE_TAG_WHITENING,
    W_TAG_FILE,
    TAG_NORMS_FILE,
    TAG_TRANSFORM_TYPE,
    ROOT_DIR
)

# Constants
DEFAULT_K = 100.0

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.drop_duplicates(subset=['appid'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def parse_tags(df):
    """
    Parses tags from the dataframe and returns:
    - sparse_counts: scipy.sparse.csr_matrix of tag counts
    - tag_to_idx: dict mapping tag name to index
    - unique_tags: list of tag names
    - appids: list of appids
    """
    print("Parsing tags...")
    all_game_tags = []
    global_tags = set()
    
    for tag_str in tqdm(df['tags'], desc="Scanning tags"):
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
            all_game_tags.append({})
            continue
        try:
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                all_game_tags.append(tags_dict)
                global_tags.update(tags_dict.keys())
            else:
                all_game_tags.append({})
        except:
            all_game_tags.append({})
            
    unique_tags = sorted(list(global_tags))
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    num_tags = len(unique_tags)
    num_games = len(all_game_tags)
    
    row_ind = []
    col_ind = []
    data = []
    
    for i, tags in enumerate(tqdm(all_game_tags, desc="Building matrix")):
        for t, c in tags.items():
            if t in tag_to_idx:
                row_ind.append(i)
                col_ind.append(tag_to_idx[t])
                data.append(c)
                
    sparse_counts = sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_games, num_tags), dtype=np.float32)
    
    return sparse_counts, tag_to_idx, unique_tags, df['appid'].values

def calculate_moments(counts_dense, threshold=1000, original_votes=None):
    """
    Calculates Global Mean (G) and Covariance (Sigma) from reliable games.
    If original_votes is provided, uses it to filter reliable games.
    Otherwise uses current sum of counts.
    """
    if original_votes is None:
        total_votes = counts_dense.sum(axis=1)
    else:
        total_votes = original_votes

    # Filter reliable games
    mask = total_votes > threshold
    if not np.any(mask):
        mask = total_votes > 0
    
    reliable_counts = counts_dense[mask]
    
    # Convert to profiles (probabilities)
    row_sums = reliable_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    profiles = reliable_counts / row_sums
    
    # Mean
    mean_profile = profiles.mean(axis=0)
    
    # Covariance
    # rowvar=False because rows are observations (games)
    cov_matrix = np.cov(profiles, rowvar=False)
    
    return mean_profile, cov_matrix

def iterative_em_imputation(sparse_counts, max_iter=TAG_EM_ITERATIONS):
    """
    Performs Iterative EM Imputation.
    
    Initial State: Calculate mean and covariance from games with >1000 votes.
    EM Loop:
        - E-Step: Impute counts for tags outside top 20 using current covariance.
        - M-Step: Re-calculate global mean and covariance based on augmented counts.
    """
    print(f"Starting Iterative EM Imputation ({max_iter} iterations)...")
    
    # Working with dense matrix
    augmented_counts = sparse_counts.toarray()
    original_counts_sparse = sparse_counts.copy() # Keep original to know what was observed
    # Keep the top 20 (Observed) fixed, and update the rest.
    # "Top 20" logic is per game.
    
    num_games, num_tags = augmented_counts.shape
    original_votes = sparse_counts.sum(axis=1).A.flatten()
    
    # Let's pre-calculate the "Observed" mask (Top 20 indices) for each game.
    print("Identifying Top 20 tags per game...")
    top_20_masks = np.zeros((num_games, num_tags), dtype=bool)
    
    for i in range(num_games):
        row = augmented_counts[i]
        # Get indices of top 20
        # argsort puts smallest first, so take last 20
        sorted_indices = np.argsort(row)
        top_indices = sorted_indices[-20:]
        # If a game has < 20 tags, all existing are "Top".
        # We only mark positive counts as "Top".
        real_top_indices = [idx for idx in top_indices if row[idx] > 0]
        top_20_masks[i, real_top_indices] = True

    # Initial M-Step
    print("Initial M-Step...")
    G, Sigma = calculate_moments(augmented_counts, threshold=1000, original_votes=original_votes)
    
    # Regularization for inversion
    ridge = TAG_EM_RIDGE
    
    for iteration in range(max_iter):
        print(f"EM Iteration {iteration + 1}/{max_iter}...")
        
        # E-Step: Impute
        # We iterate games
        
        # We can't vectorize easily because Sigma_XX varies.
        # But we can try to optimize.
        
        updates = 0
        
        for i in tqdm(range(num_games), desc="E-Step"):
            # Impute counts for tags outside the top 20. This is necessary 
            # because values after the 20th are censored. If there are
            # fewer than 20 tags, we do not impute anything, since all
            # tags are observed.
            
            # If there are fewer than 20 observed tags, skip
            if np.sum(top_20_masks[i]) < 20:
                continue

            row = augmented_counts[i]
            
            # Indices of "Observed" (Top 20) -> X
            idx_X = np.where(top_20_masks[i])[0]
            
            if len(idx_X) == 0:
                continue
                
            # Indices of "Unobserved" (Rest) -> Y
            idx_Y = np.setdiff1d(np.arange(num_tags), idx_X)
            
            if len(idx_Y) == 0:
                continue
            
            # Cap
            vals = row[idx_X]
            if len(vals) >= 20:
                cap = np.partition(vals, len(vals)-20)[len(vals)-20] # 20th largest
            else:
                cap = 0 # this should never be reached due to earlier checks
                
            # Conditional Expectation
            # P_Y_pred = mu_Y + Sigma_YX * Sigma_XX^-1 * (P_X - mu_X)
            
            # Slice Covariance
            # Sigma_XX: (len_X, len_X)
            S_XX = Sigma[np.ix_(idx_X, idx_X)]
            # Sigma_YX: (len_Y, len_X)
            S_YX = Sigma[np.ix_(idx_Y, idx_X)]
            
            # Invert S_XX
            # Add ridge
            S_XX_reg = S_XX + np.eye(len(idx_X)) * ridge
            try:
                # Use solve instead of inv for stability: S_XX_inv * vec -> solve(S_XX, vec)
                # We need S_YX @ S_XX_inv @ diff
                # Let Z = S_XX_inv @ diff -> solve(S_XX, diff)
                pass
            except np.linalg.LinAlgError:
                continue
            
            # Compute Profiles P_X
            current_sum = row.sum()
            if current_sum == 0: continue
            P_X = row[idx_X] / current_sum
            
            # Means
            mu_X = G[idx_X]
            mu_Y = G[idx_Y]
            
            diff_X = P_X - mu_X
            
            # Solve
            # Z = S_XX^-1 * diff_X
            try:
                Z = np.linalg.solve(S_XX_reg, diff_X)
            except:
                continue
                
            diff_Y = S_YX @ Z
            
            P_Y_pred = mu_Y + diff_Y
            
            # Convert to counts
            # We scale P_Y_pred to match the scale of P_X?
            # C_Y = P_Y_pred * (sum_C_X / sum_P_X)
            # sum_C_X = row[idx_X].sum()
            # sum_P_X = P_X.sum()
            # This logic assumes the profile shape is correct relative to the observed part.
            
            sum_C_X = row[idx_X].sum()
            sum_P_X = P_X.sum()
            
            if sum_P_X <= 1e-9:
                continue
                
            estimated_counts = P_Y_pred * (sum_C_X / sum_P_X)
            
            # Clip negative
            estimated_counts = np.maximum(estimated_counts, 0)
            
            # Apply Cap
            estimated_counts = np.minimum(estimated_counts, cap)
            
            # Update augmented_counts
            # We only update Y indices
            augmented_counts[i, idx_Y] = estimated_counts
            
        # M-Step
        print("M-Step: Updating moments...")
        G, Sigma = calculate_moments(augmented_counts, threshold=1000, original_votes=original_votes)
        
    return augmented_counts, G

def optimize_k_stochastic(augmented_counts, original_counts_sparse, G_prior):
    """
    Solves for K that minimizes SSE between regularized synthetic profile and 'True' flattened profile.
    
    Inputs:
    - augmented_counts: Dense matrix after EM Imputation.
    - original_counts_sparse: Sparse matrix of original counts (to get reliable games and total_votes).
    - G_prior: Global prior vector.
    """
    print("Optimizing K with Stochastic Path Optimization...")
    
    total_votes_orig = original_counts_sparse.sum(axis=1).A.flatten()
    
    # 1. Identify Reliable Games (Original > 1000 votes)
    reliable_indices = np.where(total_votes_orig >= 1000)[0]
    
    if len(reliable_indices) == 0:
        print("Warning: No reliable games (>1000 votes). Using top 1000.")
        reliable_indices = np.argsort(total_votes_orig)[-1000:]
        
    print(f"Pool of reliable games: {len(reliable_indices)}")
    
    # 2. Draw sample sizes n from actual distribution of total tag counts
    # We sample n from the FULL dataset's vote counts.
    # We filter out n=0.
    valid_votes = total_votes_orig[total_votes_orig > 0]
    sample_size = TAG_OPT_SAMPLE_SIZE
    
    sampled_ns = np.random.choice(valid_votes, size=sample_size, replace=True)
    
    # 3. Generate Synthetic Games
    # For each n, pick a random reliable game and simulate
    
    # Get True Profiles of reliable games (from Augmented counts)
    reliable_aug = augmented_counts[reliable_indices]
    reliable_sums = reliable_aug.sum(axis=1, keepdims=True)
    reliable_sums[reliable_sums == 0] = 1.0
    G_trues_pool = reliable_aug / reliable_sums
    
    synthetic_counts = []
    target_G_trues = []
    
    print("Generating synthetic games...")
    for n in sampled_ns:
        # Pick random reliable game
        idx = np.random.randint(len(reliable_indices))
        G_true = G_trues_pool[idx]
        
        # Multinomial sampling
        # Normalize strictly for numpy
        p = G_true.astype(np.float64)
        p = p / p.sum()
        
        c_syn = np.random.multinomial(n, p)
        
        synthetic_counts.append(c_syn)
        target_G_trues.append(G_true)
        
    C_syn = np.array(synthetic_counts)
    G_target = np.array(target_G_trues)
    Ns = sampled_ns.reshape(-1, 1)
    
    # 4. Solve for K
    def calculate_sse(k):
        if k < 0: return 1e12
        
        # (Csyn + K*Gprior) / (n + K)
        numerator = C_syn + k * G_prior
        denominator = Ns + k
        
        G_reg = numerator / denominator
        
        diff = G_reg - G_target
        sse = np.sum(diff**2)
        return sse
        
    res = minimize_scalar(calculate_sse, bounds=(0.1, 5000.0), method='bounded')
    
    if res.success:
        optimal_k = res.x
        print(f"Optimal K: {optimal_k:.4f} (SSE: {res.fun:.4f})")
        return optimal_k
    else:
        print(f"Optimization failed. Using default K={DEFAULT_K}")
        return DEFAULT_K

def apply_tag_transform(augmented_counts, prior_G, original_total_votes, K, transform_type='anscombe'):
    """
    Applies Bayesian regularization and then the selected Transformation.
    Ensures that a zero-tag game (after regularization) results in a zero vector.
    """
    print(f"Applying Bayesian regularization and {transform_type} transform...")
    
    # 1. Apply Bayesian Regularization to the counts first
    # Formula: (C + K*G) / (N + K)
    N = original_total_votes.reshape(-1, 1)
    
    # Broadcase N and K to match augmented_counts shape
    regularized_profiles = (augmented_counts + K * prior_G) / (N + K)
    
    # 2. Apply Transformation
    if transform_type == 'anscombe':
        # Apply transform to regularized profile
        # Note: Profiles sum to 1, so we scale them back to an 'average' count scale 
        # to make the Anscombe transform meaningful (it expects Poisson-like counts)
        avg_n = np.mean(original_total_votes)
        V = 2 * np.sqrt(regularized_profiles * avg_n + 0.375)
        # Normalize to sum to 1 to capture the shape
        V = V / V.sum(axis=1, keepdims=True)

        V_prior = 2 * np.sqrt(prior_G * avg_n + 0.375)
        V_prior = V_prior / V_prior.sum()
        
    elif transform_type == 'clr':
        # Centered Log-Ratio: log(x / geometric_mean(x))
        # Regularization ensures values are > 0 if prior_G > 0
        log_v = np.log(regularized_profiles + 1e-9)
        gm_log = log_v.mean(axis=1, keepdims=True)
        V = log_v - gm_log
        
        log_prior = np.log(prior_G + 1e-9)
        gm_log_prior = log_prior.mean()
        V_prior = log_prior - gm_log_prior
        
    else: # 'none' or Identity
        V = regularized_profiles
        V_prior = prior_G
    
    # 3. Center the vectors
    # This ensures the "regularizing point" (prior) becomes the origin.
    final_vectors = V - V_prior
    
    return final_vectors

def whiten(vectors, variance_threshold=0.80):
    """
    Whitening using PCA-based dimensionality reduction followed by ZCA rotation.
    Reduces memory footprint and eliminates singular/noisy dimensions.
    Dimensionality is chosen to retain specified proportion of variance.
    """
    print(f"Whitening vectors (PCA-ZCA with variance threshold {variance_threshold:.1%})...")
    n_games = vectors.shape[0]
    
    # M is the second moment matrix (uncentered covariance)
    M = np.dot(vectors.T, vectors) / n_games
    U, S, Vt = np.linalg.svd(M)
    
    # Compute cumulative explained variance
    cumvar = np.cumsum(S) / np.sum(S)
    n_components = np.argmax(cumvar >= variance_threshold) + 1
    if n_components <= 0:
        n_components = max(1, int(variance_threshold * len(S)))  # fallback
    actual_n = min(n_components, np.sum(S > 1e-9))
    
    print(f"Keeping top {actual_n} components (explaining {np.sum(S[:actual_n])/np.sum(S):.2%} variance)")
    
    U_reduced = U[:, :actual_n]
    S_reduced = S[:actual_n]
    
    # PCA Whitening Matrix: U_reduced @ diag(1/sqrt(S_reduced))
    # ZCA Whitening Matrix would be U_reduced @ diag(1/sqrt(S_reduced)) @ U_reduced.T
    # But we want to return whitened vectors of shape (n_games, actual_n)
    # to save memory as specified in orientation.md
    
    W = np.dot(U_reduced, np.diag(1.0 / np.sqrt(S_reduced + 1e-6)))
    whitened = np.dot(vectors, W)
    
    # Also return the full projection matrix for query transformation
    return whitened, W

def generate_tag_vectors(csv_path, output_vectors=None, output_constants=None, output_norms=None, w_tag_path=None):
    # Use defaults from constants if not provided - now pointing to data/production/
    if output_vectors is None:
        output_vectors = os.path.join(ROOT_DIR, "data", "production", "steam_tag_vectors.npy")
    if output_constants is None:
        output_constants = os.path.join(ROOT_DIR, "data", "production", "regularization_constants.json")
    if output_norms is None:
        output_norms = os.path.join(ROOT_DIR, "data", "production", "tag_vectors_norms.npy")
    if w_tag_path is None:
        w_tag_path = os.path.join(ROOT_DIR, "data", "production", "w_tag.npy")
    
    df = load_data(csv_path)
    sparse_counts, tag_to_idx, unique_tags, appids = parse_tags(df)
    
    original_total_votes = np.array(sparse_counts.sum(axis=1)).flatten()
    
    # 1. EM Imputation
    augmented_counts, G_final = iterative_em_imputation(sparse_counts, max_iter=TAG_EM_ITERATIONS)
    
    # 2. Stochastic Path Optimization
    K = optimize_k_stochastic(augmented_counts, sparse_counts, G_final)
    
    # 4. Transform + Dampening
    transformed_vectors = apply_tag_transform(augmented_counts, G_final, original_total_votes, K, transform_type=TAG_TRANSFORM_TYPE)
    
    # 5. Whiten
    if USE_TAG_WHITENING:
        whitened_vectors, W = whiten(transformed_vectors, variance_threshold=0.80)
    else:
        print("Skipping whitening (Identity transform)...")
        whitened_vectors = transformed_vectors
        # Create Identity matrix of appropriate size
        dim = transformed_vectors.shape[1]
        W = np.eye(dim)
    
    print(f"Saving vectors to {output_vectors}...")
    np.save(output_vectors, whitened_vectors.astype(np.float16))

    norms_path = output_norms if output_norms else TAG_NORMS_FILE
    print(f"Saving tag vector norms to {norms_path}...")
    tag_norms = np.linalg.norm(whitened_vectors.astype(np.float32), axis=1).astype(np.float16)
    np.save(norms_path, tag_norms)

    final_w_path = w_tag_path if w_tag_path else W_TAG_FILE
    print(f"Saving whitening matrix to {final_w_path}...")
    np.save(final_w_path, W.astype(np.float16))

    # Run distribution analysis
    try:
        from research.analyze_vector_distributions import analyze_distribution
        analyze_distribution(whitened_vectors, "Steam Tag Vectors")
    except ImportError:
        print("Warning: could not import analyze_distribution from research.analyze_vector_distributions")
    
    # Calculate DOT_PRODUCT_LAMBDA
    lambda_val = calculate_dot_product_lambda(whitened_vectors)
    
    # Save Constants
    reg_constants = {}
    if os.path.exists(output_constants):
        try:
            with open(output_constants, "r") as f:
                reg_constants = json.load(f)
        except:
            pass
            
    reg_constants["TAG_VECTOR_K"] = float(K)
    reg_constants["DOT_PRODUCT_LAMBDA"] = float(lambda_val)
    
    print(f"Saving constants to {output_constants}...")
    with open(output_constants, "w") as f:
        json.dump(reg_constants, f, indent=4)
        
    # Add parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from common.constants import TAG_NAMES_FILE
    
    print(f"Saving master tag list to {TAG_NAMES_FILE}...")
    with open(TAG_NAMES_FILE, 'w') as f:
        json.dump(unique_tags, f, indent=4)
        
    return whitened_vectors, appids

def calculate_dot_product_lambda(vectors):
    """
    Calculates DOT_PRODUCT_LAMBDA by fitting a Chi-distribution to the lengths 
    of 'low-tag' vectors and taking the 95th percentile.
    """
    print("Calculating DOT_PRODUCT_LAMBDA via Chi-distribution fit...")
    # Calculate L2 norms (lengths)
    lengths = np.linalg.norm(vectors, axis=1)
    
    # Filter for non-zero vectors in the 'noise' range
    subset_mask = (lengths > 1e-6) & (lengths <= CHI_FIT_NORM_THRESHOLD)
    subset_lengths = lengths[subset_mask]
    
    if len(subset_lengths) > 10: # Ensure enough samples for a fit
        try:
            # Fit Chi-distribution
            df, loc, scale = chi.fit(subset_lengths)
            # Calculate 95th percentile
            data_driven_lambda = chi.ppf(CHI_FIT_PERCENTILE, df, loc, scale)
            print(f"Fitted Chi: df={df:.4f}, loc={loc:.4f}, scale={scale:.4f}")
        except Exception as e:
            print(f"Warning: Chi-fit failed ({e}). Falling back to variance.")
            data_driven_lambda = np.var(subset_lengths)
    else:
        print(f"Warning: Too few vectors in range (0, {CHI_FIT_NORM_THRESHOLD}]. Using default lambda=1.0")
        data_driven_lambda = 1.0

    print(f"Recommended Lambda: {data_driven_lambda:.4f}")
    return data_driven_lambda

class TagSearchEngine:
    def __init__(self, vectors, appids):
        self.vectors = vectors
        self.appids = appids
        self.appid_to_idx = {appid: i for i, appid in enumerate(appids)}
        
    def search(self, query_vector, k=10):
        scores = np.dot(self.vectors, query_vector)
        top_indices = np.argsort(scores)[-k:][::-1]
        return [(self.appids[i], scores[i]) for i in top_indices]

    def get_vector(self, appid):
        idx = self.appid_to_idx.get(appid)
        if idx is not None:
            return self.vectors[idx]
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate tag vectors and calculate constants")
    parser.add_argument("csv", default="data/pipeline_games_clean.csv", nargs='?', help="Path to cleaned games CSV")
    parser.add_argument("--output", default=None, help="Path to output vectors (.npy)")
    parser.add_argument("--constants", default=None, help="Path to output constants (.json)")
    parser.add_argument("--norms", default=None, help="Path to output norms (.npy)")
    parser.add_argument("--w_tag", default=None, help="Path to output whitening matrix (.npy)")
    args = parser.parse_args()
        
    # Determine output paths from constants if not provided
    output_vectors = args.output if args.output else os.path.join(ROOT_DIR, "data", "production", "steam_tag_vectors.npy")
    output_constants = args.constants if args.constants else os.path.join(ROOT_DIR, "data", "production", "regularization_constants.json")
    output_norms = args.norms if args.norms else os.path.join(ROOT_DIR, "data", "production", "tag_vectors_norms.npy")
    w_tag_path = args.w_tag if args.w_tag else os.path.join(ROOT_DIR, "data", "production", "w_tag.npy")
        
    vectors, appids = generate_tag_vectors(
        args.csv, 
        output_vectors=output_vectors, 
        output_constants=output_constants, 
        output_norms=output_norms,
        w_tag_path=w_tag_path
    )
    
    searcher = TagSearchEngine(vectors, appids)
    v1 = searcher.get_vector(10) # CS
    v2 = searcher.get_vector(240) # CS:S
    if v1 is not None and v2 is not None:
        sim = np.dot(v1, v2)
        print(f"Similarity CS (10) vs CS:Source (240): {sim:.4f}")