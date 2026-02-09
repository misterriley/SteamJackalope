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
    if ignore_zeros:
        # Use a small threshold to handle numerical noise in dense/whitened vectors
        subset = x[np.abs(x) > 1e-5]
        if len(subset) == 0:
            # Fallback to non-ignored if everything is near zero
            subset = x
            
        mean = np.mean(subset)
        std = np.std(subset)
    else:
        mean = np.mean(x)
        std = np.std(x)
    
    z = (x - mean) / (std if std > EPSILON else 1.0)
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
