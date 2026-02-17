import numpy as np
from common.constants import EPSILON, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX

def to_z(x, ignore_zeros=False):
    """
    Calculates the z-score of a numerical array.
    
    Args:
        x (np.array): Input data.
        ignore_zeros (bool): If True, values near zero are ignored when calculating mean and std.
                            This is useful for sparse similarity distributions.

    Returns:
        np.array: Z-scored data.
    """
    # Convert to numpy array to ensure dtype parameter works
    x_array = np.asarray(x)
    
    if ignore_zeros:
        # Use a small threshold to handle numerical noise in dense/whitened vectors
        subset = x_array[np.abs(x_array) > 1e-5]
        if len(subset) == 0:
            # Fallback to non-ignored if everything is near zero
            subset = x_array
            
        mean = np.mean(subset, dtype=np.float64)
        std = np.std(subset, dtype=np.float64)
    else:
        mean = np.mean(x_array, dtype=np.float64)
        std = np.std(x_array, dtype=np.float64)
    
    z = (x_array - mean) / (std if std > EPSILON else 1.0)
    return z

def calculate_linear_scores(
    z_quality, 
    z_date, 
    z_pop, 
    z_playtime, 
    z_difficulty,
    tag_vectors, 
    tag_norms, 
    beta_tag,
    weights, 
    intercept,
    tag_scaling_factor, 
    dot_product_lambda,
    z_clamp_min, 
    z_clamp_max
):
    """
    Unified linear scoring function for Taste DNA parity.
    This is the single source of truth for both the Solver preview and Backend recommender.
    """
    # 1. Apply Clamping to Metadata
    q = np.clip(z_quality, z_clamp_min, z_clamp_max)
    d = np.clip(z_date, z_clamp_min, z_clamp_max)
    p = np.clip(z_pop, z_clamp_min, z_clamp_max)
    l = np.clip(z_playtime, z_clamp_min, z_clamp_max)
    diff = np.clip(z_difficulty, z_clamp_min, z_clamp_max)
    
    # 2. Tag Scoring: dot(U / (||U|| + lambda) * Scale, beta_absolute)
    # beta_tag should already be the absolute coefficient vector (unit * tag_match_norm)
    dot_products = np.dot(tag_vectors, beta_tag)
    # Ensure tag_norms is a vector matching tag_vectors length
    denom = tag_norms.reshape(-1) + dot_product_lambda
    tag_contrib = (dot_products / denom) * tag_scaling_factor
    
    # 3. Summation: Intercept + sum(beta_i * feature_i)
    scores = (
        q * weights.get('quality', 0.0) +
        d * weights.get('age', 0.0) +
        p * weights.get('popularity', 0.0) +
        l * weights.get('length', 0.0) +
        diff * weights.get('difficulty', 0.0) +
        tag_contrib +
        intercept
    )
    return scores

def calculate_hybrid_score(
    z_semantic, w_semantic,
    z_tag, w_tag,
    z_spps, w_spps,
    z_date, w_date,
    z_pop, w_pop,
    z_length, w_length,
    z_difficulty, w_difficulty
):
    """
    Calculates the final hybrid score by blending multiple components.
    
    Args:
        z_semantic (np.array): Semantic similarity z-scores.
        w_semantic (float): Weight for semantic component.
        z_tag (np.array): Tag similarity z-scores.
        w_tag (float): Weight for tag component.
        z_spps (np.array): Quality/Rating z-scores.
        w_spps (float): Weight for quality component.
        z_date (np.array): Release date z-scores.
        w_date (float): Adjusted weight for age preference.
        z_pop (np.array): Popularity z-scores.
        w_pop (float): Adjusted weight for popularity preference.
        z_length (np.array): Game length z-scores.
        w_length (float): Adjusted weight for length preference.
        z_difficulty (np.array): Difficulty z-scores.
        w_difficulty (float): Adjusted weight for difficulty preference.
        
    Returns:
        np.array: Final blended scores.
    """
    raw_score = (
        z_semantic * w_semantic +
        z_tag * w_tag +
        z_spps * w_spps +
        z_date * w_date +
        z_pop * w_pop +
        z_length * w_length +
        z_difficulty * w_difficulty
    )

    total_weight = (
        abs(w_semantic) + 
        abs(w_tag) + 
        abs(w_spps) + 
        abs(w_date) + 
        abs(w_pop) + 
        abs(w_length) +
        abs(w_difficulty)
    )

    if total_weight == 0:
        return np.zeros_like(z_semantic)
    
    return raw_score / total_weight

def calculate_personalized_quality(q_global, p_plus_playtime):
    """
    Adjusts global quality scores (probits) to reflect the expected experience
    at a specific playtime, based on the biased mean of the underlying normal distribution.
    
    This handles the case where we sample from N(Q, 1) but with a biased sampling 
    probability p_plus_playtime instead of the natural probability Phi(Q).

    Args:
        q_global (np.ndarray): The raw probit quality scores (mu/mean of the distribution).
        p_plus_playtime (np.ndarray): The predicted probability of a positive review 
                                      at a specific playtime (from kernel smoothing).

    Returns:
        np.ndarray: The Personalized Quality score (Expected Experience).
    """
    from scipy.stats import norm
    
    # phi(Q) and Phi(Q)
    # Using float64 for intermediate steps to avoid overflow in tails
    q_64 = q_global.astype(np.float64)
    pdf_q = norm.pdf(q_64)
    cdf_q = norm.cdf(q_64)
    sf_q = norm.sf(q_64) # Survival function (1 - Phi), more stable for high Q
    
    # Calculate the shift factor: [p+/Phi - (1-p+)/(1-Phi)]
    # We use a small epsilon to avoid division by zero in extreme tails
    eps = 1e-12
    shift = (p_plus_playtime / (cdf_q + eps)) - ((1.0 - p_plus_playtime) / (sf_q + eps))
    
    # E[S] = Q + phi(Q) * shift
    personalized_q = q_64 + pdf_q * shift
    
    return personalized_q.astype(np.float32)

def calculate_dot_product_lambda(vectors):
    """
    Calculates DOT_PRODUCT_LAMBDA by fitting a Chi-distribution to the lengths 
    of 'low-tag' vectors and taking the 95th percentile.
    """
    from common.constants import CHI_FIT_NORM_THRESHOLD, CHI_FIT_PERCENTILE
    from scipy.stats import chi
    
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

def safe_save_npy(path, data):
    """
    Saves a NumPy array to a file safely, handling Windows file locking 
    (e.g., when the file is memory-mapped by another process or locked by OneDrive).
    """
    import os
    import time
    import shutil
    
    # Ensure path ends in .npy
    if not path.endswith('.npy'):
        path += '.npy'
        
    # Create directory if it doesn't exist
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # We save to a unique temp file to avoid collisions and allow multiple attempts
    temp_path = path + f".{os.getpid()}.tmp"
    actual_temp_file = temp_path + ".npy"
    
    try:
        np.save(temp_path, data)
    except Exception as e:
        print(f"CRITICAL: Failed to write initial temp file to {temp_path}: {e}")
        raise e
    
    max_retries = 10
    retry_delay = 1.0 # seconds
    
    for attempt in range(max_retries):
        try:
            # 1. Try direct replace (atomic on many systems)
            if os.path.exists(path):
                os.replace(actual_temp_file, path)
            else:
                os.rename(actual_temp_file, path)
            return
        except OSError as e:
            # WinError 32: File used by another process
            # WinError 5: Access is denied (often occurs if file is memory-mapped)
            
            # 2. Try the rename-to-garbage trick (sometimes works for memory-mapped files)
            garbage_path = path + f".old.{int(time.time())}.{attempt}"
            try:
                os.rename(path, garbage_path)
                # Success! Now the target path is free.
                os.rename(actual_temp_file, path)
                # Try to clean up garbage, but don't fail if we can't
                try:
                    os.remove(garbage_path)
                except:
                    pass
                return
            except OSError:
                # Both direct replace and rename-trick failed. 
                # This usually means a hard lock (like a memory map in an active process).
                if attempt < max_retries - 1:
                    if attempt == 0:
                        print(f"Warning: {path} is locked. Retrying ({attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    print(f"CRITICAL: Failed to update {path} after {max_retries} attempts.")
                    print(f"This is likely because the file is locked by the FastAPI server or OneDrive.")
                    print(f"ACTION REQUIRED: Please stop the FastAPI server and try again.")
                    print(f"The new data has been saved to: {actual_temp_file}")
                    raise e
