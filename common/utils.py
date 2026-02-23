import numpy as np
from common.constants import EPSILON, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX, SOFTMIN_TEMPERATURE

def to_z(x, ignore_zeros=False):
    """
    Calculates the z-score of a numerical array.
    """
    x_array = np.asarray(x)
    if ignore_zeros:
        subset = x_array[np.abs(x_array) > 1e-5]
        if len(subset) == 0:
            subset = x_array
        mean = np.mean(subset, dtype=np.float64)
        std = np.std(subset, dtype=np.float64)
    else:
        mean = np.mean(x_array, dtype=np.float64)
        std = np.std(x_array, dtype=np.float64)
    z = (x_array - mean) / (std if std > EPSILON else 1.0)
    return z

def calculate_linear_scores(
    z_quality, z_date, z_pop, z_playtime, z_difficulty, z_price,
    tag_vectors, tag_norms, beta_tag,
    weights,
    tag_scaling_factor,
    dot_product_lambda,
    z_semantic=None,
    w_semantic=0.0,
    z_clamp_min=-3.0,
    z_clamp_max=3.0,
    dna_scaling_factor=1.0,
    intercept=5.0,
    tag_sim=None
):
    """
    Unified linear scoring function for Taste DNA parity.
    This is the single source of truth for both the Solver preview and Backend recommender.
    
    All input weights (in 'weights' and 'w_semantic') should be in the TARGET scale
    (e.g., UI scale where 5.0 is neutral and 3.0 is a typical max weight).
    """
    # 1. Apply Clamping to Metadata
    q = np.clip(z_quality, z_clamp_min, z_clamp_max)
    d = np.clip(z_date, z_clamp_min, z_clamp_max)
    p = np.clip(z_pop, z_clamp_min, z_clamp_max)
    l = np.clip(z_playtime, z_clamp_min, z_clamp_max)
    diff = np.clip(z_difficulty, z_clamp_min, z_clamp_max)
    pr = np.clip(z_price, z_clamp_min, z_clamp_max)
    
    # 2. Tag Scoring: dot(U / (||U|| + lambda) * Scale, beta_unit)
    if tag_sim is None:
        beta_tag_arr = np.asarray(beta_tag, dtype=np.float32)
        dot_products = np.dot(tag_vectors.astype(np.float32), beta_tag_arr)
        denom = tag_norms.astype(np.float32).reshape(-1) + dot_product_lambda
        tag_sim = (dot_products / denom) * tag_scaling_factor
    
    # 3. Semantic Component
    sem_contrib = (z_semantic * w_semantic) if z_semantic is not None else 0.0

    # 4. Summation: sum(weight_i * feature_i) + intercept
    scores = (
        q * weights.get('quality', 0.0) +
        d * weights.get('age', 0.0) +
        p * weights.get('popularity', 0.0) +
        l * weights.get('length', 0.0) +
        diff * weights.get('difficulty', 0.0) +
        pr * weights.get('price', 0.0) +
        (tag_sim * weights.get('tag_match', 0.0)) +
        sem_contrib
    )
    # Divide by scaling factor to map back to original 0-10 scale
    # Only scale the deviations from the intercept.
    return (scores / dna_scaling_factor) + intercept

def calculate_hybrid_score(
    z_semantic, w_semantic,
    z_tag, w_tag,
    z_spps, w_spps,
    z_date, w_date,
    z_pop, w_pop,
    z_length, w_length,
    z_difficulty, w_difficulty,
    z_price, w_price
):
    """
    Calculates the final hybrid score for recommender mode (Manual Sliders).
    """
    return (
        (z_semantic * w_semantic) +
        (z_tag * w_tag) +
        (z_spps * w_spps) +
        (z_date * w_date) +
        (z_pop * w_pop) +
        (z_length * w_length) +
        (z_difficulty * w_difficulty) +
        (z_price * w_price) +
        5.0 # Neutral Anchor
    )

def ensure_python_types(obj):
    """
    Recursively converts NumPy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: ensure_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_python_types(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return ensure_python_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def calculate_personalized_quality(q_global, p_plus_playtime):
    """
    Adjusts global quality scores (probits) to reflect the expected experience
    at a specific playtime.
    """
    from scipy.stats import norm
    q_64 = q_global.astype(np.float64)
    pdf_q = norm.pdf(q_64)
    cdf_q = norm.cdf(q_64)
    sf_q = norm.sf(q_64)
    eps = 1e-12
    shift = (p_plus_playtime / (cdf_q + eps)) - ((1.0 - p_plus_playtime) / (sf_q + eps))
    personalized_q = q_64 + pdf_q * shift
    return personalized_q.astype(np.float32)

def calculate_dot_product_lambda(vectors):
    """
    Calculates DOT_PRODUCT_LAMBDA via Chi-distribution fit.
    """
    from common.constants import CHI_FIT_NORM_THRESHOLD, CHI_FIT_PERCENTILE
    from scipy.stats import chi
    lengths = np.linalg.norm(vectors, axis=1)
    subset_mask = (lengths > 1e-6) & (lengths <= CHI_FIT_NORM_THRESHOLD)
    subset_lengths = lengths[subset_mask]
    if len(subset_lengths) > 10:
        df, loc, scale = chi.fit(subset_lengths)
        data_driven_lambda = chi.ppf(CHI_FIT_PERCENTILE, df, loc, scale)
    else:
        data_driven_lambda = 1.0
    return data_driven_lambda

def safe_save_npy(path, data):
    """
    Saves a NumPy array to a file safely handling Windows file locking.
    """
    import os
    import time
    if not path.endswith('.npy'):
        path += '.npy'
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    temp_path = path + f".{os.getpid()}.tmp"
    actual_temp_file = temp_path + ".npy"
    np.save(temp_path, data)
    max_retries = 10
    retry_delay = 1.0
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                os.replace(actual_temp_file, path)
            else:
                os.rename(actual_temp_file, path)
            return
        except OSError:
            garbage_path = path + f".old.{int(time.time())}.{attempt}"
            try:
                os.rename(path, garbage_path)
                os.rename(actual_temp_file, path)
                try:
                    os.remove(garbage_path)
                except:
                    pass
                return
            except OSError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

def softmin_blend(signals: list, temperature: float = SOFTMIN_TEMPERATURE):
    """
    Blends multiple similarity signals using a softmin-weighted average.
    Signals are expected to be NumPy arrays of the same shape.
    This creates 'consensus' logic: the result is heavily weighted towards the lowest score.
    """
    if not signals:
        return 0.0
    if len(signals) == 1:
        return signals[0]
        
    stack = np.stack(signals, axis=0) # (num_signals, num_games)
    
    # Calculate weights: w_i = exp(-s_i / T) / sum(exp(-s_j / T))
    # Using log-sum-exp trick for stability
    scaled = -stack / temperature
    max_val = np.max(scaled, axis=0)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals, axis=0)
    
    # Blended similarity: sum(s_i * w_i)
    return np.sum(stack * weights, axis=0)
