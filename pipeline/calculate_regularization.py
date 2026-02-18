import pandas as pd
import numpy as np
import json
import os
import sys
import ast
from collections import Counter
from scipy.optimize import minimize_scalar

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    REG_RATE_MIN_REVIEWS_THRESHOLD,
    REG_TAG_SAMPLE_SIZE,
    REG_TAG_MIN_VOTES_THRESHOLD,
    REG_PLAYTIME_REVIEWS_THRESHOLD,
    REG_PLAYTIME_SAFETY_FACTOR,
    REG_PLAYTIME_C_MIN,
    REG_PLAYTIME_C_MAX,
    QUALITY_SCORE_S_CONST
)

from scipy.stats import norm

def calculate_rating_anchors(df, s, a):
    """
    Calculates mapping coefficients m and c to map Probit scores to 0-10 ratings.
    Anchors are derived from the theoretical absolute extremes of the current dataset.
    """
    p = df['positive'].fillna(0).values
    n = df['negative'].fillna(0).values
    
    # Calculate Probit Q for all games
    prob = (p + s * a) / (p + n + s)
    q = norm.ppf(np.clip(prob, 1e-6, 1-1e-6))
    
    q_max = np.max(q)
    q_min = np.min(q)
    
    # Calculate Hard Labels (Personalized Q) for the absolute extremes
    # p+ = 1 for the best theoretical experience
    q_pers_max = q_max + norm.pdf(q_max) / norm.cdf(q_max)
    # p+ = 0 for the worst theoretical experience
    q_pers_min = q_min - norm.pdf(q_min) / norm.sf(q_min)
    
    # Solve for Rating = m * Q_pers + c
    # 10 = m * q_pers_max + c
    # 0 = m * q_pers_min + c
    m = 10.0 / (q_pers_max - q_pers_min)
    c = -m * q_pers_min
    
    return float(m), float(c)

def calculate_global_positive_rate(df):
    """
    Calculates the global positive review rate using games with at least a 
    minimum number of reviews to provide a stable and representative prior.
    """
    # Filter for reliable games to get a stable mean
    mask = (df['positive'] + df['negative']) >= REG_RATE_MIN_REVIEWS_THRESHOLD
    df_reliable = df[mask].copy()
    
    if df_reliable.empty:
        # Fallback to all reviews if no games meet the threshold
        pos = df['positive'].fillna(0).sum()
        neg = df['negative'].fillna(0).sum()
        return float(pos / (pos + neg)) if (pos + neg) > 0 else 0.86
    
    # Calculate average of game-level rates
    rates = df_reliable['positive'] / (df_reliable['positive'] + df_reliable['negative'])
    return float(rates.mean())

def solve_tag_vector_k(df):
    """
    Optimizes the Bayesian smoothing constant K for tag vectors using 
    cross-validation on a sample of games.
    """
    print("Parsing tags for K optimization...")
    all_game_tags = []
    global_vote_counts = Counter()
    
    for tag_str in df['tags']:
        if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
            continue
        try:
            tags_dict = ast.literal_eval(tag_str)
            if isinstance(tags_dict, dict):
                all_game_tags.append(tags_dict)
                global_vote_counts.update(tags_dict)
            else:
                d = {t: 1 for t in tags_dict}
                all_game_tags.append(d)
                global_vote_counts.update(d)
        except:
            continue
            
    if not all_game_tags:
        return 164.1093
        
    unique_tags = sorted(global_vote_counts.keys())
    tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
    num_tags = len(unique_tags)
    
    total_global_votes = sum(global_vote_counts.values())
    G = np.array([global_vote_counts[tag] for tag in unique_tags], dtype=float) / total_global_votes
    
    # Filter for 'Reliable' games (N >= 1000)
    reliable_games = []
    for t in all_game_tags:
        total_votes = sum(t.values())
        if total_votes >= 1000:
            reliable_games.append((t, total_votes))
            
    # Select 1,000 reliable games
    if not reliable_games:
        print("Warning: No reliable games (>= 1000 votes) found. Using all available games.")
        reliable_games = [(t, sum(t.values())) for t in all_game_tags]
        
    sample_size = min(1000, len(reliable_games))
    print(f"Running Stochastic Path Augmentation on {sample_size} reliable games...")
    
    indices = np.random.choice(len(reliable_games), sample_size, replace=False)
    selected_games = [reliable_games[i] for i in indices]
    
    np.random.seed(42)
    
    # Prepare data arrays
    G_game_list = [] # True profiles
    C_syn_list = []  # Synthetic small game votes
    n_list = []      # Synthetic vote counts
    
    for tags_dict, total_votes in selected_games:
        # Create true profile G_game
        vec = np.zeros(num_tags)
        for t, v in tags_dict.items():
            vec[tag_to_idx[t]] = v
        
        G_game = vec / total_votes
        G_game_list.append(G_game)
        
        # Create Synthetic Small Game
        # Sample n votes (random between 1 and 100)
        n = np.random.randint(1, 101)
        n_list.append(n)
        
        # Sample C_syn using multinomial(n, G_game)
        C_syn = np.random.multinomial(n, G_game)
        C_syn_list.append(C_syn)
        
    G_game_arr = np.array(G_game_list)
    C_syn_arr = np.array(C_syn_list)
    n_arr = np.array(n_list).reshape(-1, 1)
    
    # Global mean G is already calculated
    
    def log_likelihood(K):
        if K < 0: return 1e12
        
        # Profile_reg = (C_syn + K * G_global) / (n + K)
        # Broadcasting: (N_games, N_tags) + scalar * (N_tags,) / (N_games, 1) + scalar
        numerator = C_syn_arr + K * G
        denominator = n_arr + K
        
        Profile_reg = numerator / denominator
        
        # Clip to avoid log(0)
        epsilon = 1e-12
        Profile_reg = np.clip(Profile_reg, epsilon, 1.0)
        
        # Loss = - sum(G_game * log(Profile_reg))
        loss = -np.sum(G_game_arr * np.log(Profile_reg))
        return loss

    res = minimize_scalar(log_likelihood, bounds=(0, 2000), method='bounded')
    return float(res.x) if res.success else 164.1093

def solve_playtime_regularization(reviews_path, threshold=REG_PLAYTIME_REVIEWS_THRESHOLD):
    """
    Calculates the regularization constant C for playtime shrinkage using Stochastic Path Analysis.
    Metric: Logarithm of the median value of playtime for positive reviews only.
    """
    if not os.path.exists(reviews_path):
        return 100.0

    print(f"Calculating playtime regularization from {reviews_path} using Stochastic Path Analysis (Threshold={threshold})...")
    
    try:
        # 1. Load Data
        df = pd.read_csv(reviews_path, usecols=['appid', 'author_playtime_forever', 'voted_up'])
        
        # 2. Filter for Positive Reviews and valid playtime
        df = df[df['voted_up'] == True]
        df = df[df['author_playtime_forever'] > 0]
        
        # 3. Group by AppID
        grouped = df.groupby('appid')['author_playtime_forever']
        
        # Get counts
        counts = grouped.count()
        
        # 4. Identify Reliable Games
        reliable_appids = counts[counts >= threshold].index
        
        if len(reliable_appids) == 0:
            print("Warning: No reliable games found for playtime regularization. Defaulting.")
            return 100.0
            
        print(f"Found {len(reliable_appids)} reliable games (>= {threshold} positive reviews).")
        
        # 5. Extract True Profiles (Log-Medians) and Raw Data for Reliable Games
        # We need the list of playtimes to bootstrap
        reliable_df = df[df['appid'].isin(reliable_appids)]
        
        # Group again
        reliable_grouped = reliable_df.groupby('appid')['author_playtime_forever']
        
        # Calculate True Log-Medians
        true_medians = reliable_grouped.median()
        true_log_medians = np.log1p(true_medians)
        
        # Calculate Global Prior (Mean of Log-Medians of Reliable Games)
        global_log_median_prior = true_log_medians.mean()
        
        print(f"Global Log-Median Prior: {global_log_median_prior:.4f}")
        
        # 6. Stochastic Sampling
        # Sample size for optimization
        opt_sample_size = 2000
        
        # Pool of valid 'n' (counts) from ALL games (not just reliable) to simulate real sparsity
        valid_counts = counts[counts > 0].values
        
        # Generate synthetic samples
        synthetic_samples = [] # (syn_log_median, true_log_median, n)
        
        # Pre-fetch playtimes for reliable games to speed up sampling
        # Dict: appid -> array of playtimes
        playtime_pool = {appid: group.values for appid, group in reliable_grouped}
        reliable_appid_list = list(reliable_appids)
        
        np.random.seed(42)
        
        print("Generating synthetic samples...")
        for _ in range(opt_sample_size):
            # Pick a random reliable game
            appid = np.random.choice(reliable_appid_list)
            true_val = true_log_medians[appid]
            playtimes = playtime_pool[appid]
            
            # Pick a random n (simulate a small game)
            n = np.random.choice(valid_counts)
            
            # Bootstrap sample n playtimes
            sampled_playtimes = np.random.choice(playtimes, size=n, replace=True)
            
            # Calculate synthetic statistic
            syn_median = np.median(sampled_playtimes)
            syn_log_median = np.log1p(syn_median)
            
            synthetic_samples.append((syn_log_median, true_val, n))
            
        # Convert to arrays for vectorization
        syn_arr = np.array([x[0] for x in synthetic_samples])
        true_arr = np.array([x[1] for x in synthetic_samples])
        n_arr = np.array([x[2] for x in synthetic_samples])
        
        # 7. Minimize SSE
        def calculate_sse(C):
            if C < 0.1: return 1e12
            
            # Regularized Estimate
            # reg = (n * syn + C * prior) / (n + C)
            reg_est = (n_arr * syn_arr + C * global_log_median_prior) / (n_arr + C)
            
            # Error
            diff = reg_est - true_arr
            sse = np.sum(diff**2)
            return sse
            
        res = minimize_scalar(calculate_sse, bounds=(0.1, 5000.0), method='bounded')
        
        if res.success:
            print(f"Optimal Playtime Regularization C: {res.x:.4f} (SSE: {res.fun:.4f})")
            return float(res.x)
        else:
            print("Optimization failed. Returning default.")
            return 100.0
            
    except Exception as e:
        print(f"Error calculating playtime regularization: {e}")
        return 100.0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate regularization constants")
    parser.add_argument("--csv", default="data/pipeline_games_clean.csv", help="Path to cleaned games CSV")
    parser.add_argument("--reviews", default="scraped_reviews.csv", help="Path to reviews CSV")
    parser.add_argument("--output", default="regularization_constants.json", help="Path to output JSON")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found. Run run_pipeline.py first.")
        return

    print(f"Loading {args.csv} for regularization calculation...")
    df = pd.read_csv(args.csv, low_memory=False)
    
    # Load existing constants to preserve them
    constants_dict = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r") as f:
                constants_dict = json.load(f)
        except:
            pass
    
    constants_dict["GLOBAL_POSITIVE_RATE"] = calculate_global_positive_rate(df)
    print(f"Calculated GLOBAL_POSITIVE_RATE: {constants_dict['GLOBAL_POSITIVE_RATE']:.4f}")
    
    # TAG_VECTOR_K is now calculated in generate_tag_vectors.py using Iterative EM Imputation
    if "TAG_VECTOR_K" not in constants_dict:
        constants_dict["TAG_VECTOR_K"] = 100.0
        
    print(f"Using placeholder/existing TAG_VECTOR_K: {constants_dict['TAG_VECTOR_K']:.4f} (will be optimized in generate_tag_vectors.py)")
    
    constants_dict["PLAYTIME_REGULARIZATION_C"] = solve_playtime_regularization(args.reviews)
    print(f"Calculated PLAYTIME_REGULARIZATION_C: {constants_dict['PLAYTIME_REGULARIZATION_C']:.4f}")
    
    # Calculate Rating Anchors (m and c)
    s = QUALITY_SCORE_S_CONST
    a = constants_dict["GLOBAL_POSITIVE_RATE"]
    m, c = calculate_rating_anchors(df, s, a)
    constants_dict["QUALITY_TO_RATING_SLOPE"] = m
    constants_dict["QUALITY_TO_RATING_INTERCEPT"] = c
    print(f"Calculated Rating Anchors: m={m:.6f}, c={c:.6f}")
    
    print(f"Saving constants to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(constants_dict, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    main()
