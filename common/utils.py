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
