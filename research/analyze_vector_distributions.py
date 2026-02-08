import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def analyze_distribution(vectors, name="vectors"):
    print(f"\n--- Analyzing {name} ---")
    
    # 1. Vector Lengths
    lengths = np.linalg.norm(vectors, axis=1)
    # Remove zeros for length analysis to avoid log(0) etc if applicable
    lengths = lengths[lengths > 1e-9]
    
    if len(lengths) == 0:
        print("No non-zero vectors found.")
        return

    # Theoretical length distribution for N(0, 1) vectors in D dimensions
    # Length of a D-dimensional vector of N(0, 1) variables follows a Chi distribution with D degrees of freedom.
    # If the vectors are whitened, they should have roughly unit variance per dimension.
    d = vectors.shape[1]
    
    # Calculate moments
    mean_len = np.mean(lengths)
    var_len = np.var(lengths)
    
    # Theoretical Chi moments for df=d
    # Normal approximation for large d: Chi(d) ~ N(sqrt(d - 0.5), 0.5)
    if d > 100:
        theo_mean = np.sqrt(d - 0.5)
        theo_var = 0.5
    else:
        theo_mean = stats.chi.mean(df=d)
        theo_var = stats.chi.var(df=d)
    
    print(f"Dimensions: {d}")
    print(f"Length Distribution:")
    print(f"  Observed Mean: {mean_len:.4f} (Theoretical Chi({d}): {theo_mean:.4f})")
    print(f"  Observed Var:  {var_len:.4f} (Theoretical Chi({d}): {theo_var:.4f})")
    
    # KS Test for lengths
    # For large d, Chi(d) is approx Normal(sqrt(d - 0.5), sqrt(0.5))
    if d > 100:
        # Scale the lengths so mean matches theoretical
        scale_est = mean_len / np.sqrt(d - 0.5)
        scaled_lengths = lengths / scale_est
        ks_stat, ks_p = stats.kstest(scaled_lengths, 'norm', args=(np.sqrt(d - 0.5), np.sqrt(0.5)))
        print(f"  KS Test vs N(sqrt({d}-0.5), 0.5) (scaled by {scale_est:.4f}): stat={ks_stat:.4f}, p={ks_p:.4g}")
    else:
        scale_est = mean_len / theo_mean
        ks_stat, ks_p = stats.kstest(lengths, 'chi', args=(d, 0, scale_est))
        print(f"  KS Test vs Chi({d}) (scaled by {scale_est:.4f}): stat={ks_stat:.4f}, p={ks_p:.4g}")

    # 2. Cosine Similarities
    # To avoid N^2, sample pairs
    num_samples = 10000
    indices1 = np.random.choice(len(vectors), num_samples)
    indices2 = np.random.choice(len(vectors), num_samples)
    
    # Ensure they are different
    mask = indices1 != indices2
    idx1, idx2 = indices1[mask], indices2[mask]
    
    v1 = vectors[idx1]
    v2 = vectors[idx2]
    
    dot_products = np.sum(v1 * v2, axis=1)
    norms1 = np.linalg.norm(v1, axis=1)
    norms2 = np.linalg.norm(v2, axis=1)
    
    # Avoid div by zero
    valid = (norms1 > 1e-9) & (norms2 > 1e-9)
    cos_sims = dot_products[valid] / (norms1[valid] * norms2[valid])
    
    # Cosine similarity of two random vectors in D dimensions (from normal)
    # follows a distribution related to Beta. 
    # Specifically, (cos_sim + 1) / 2 ~ Beta((D-1)/2, (D-1)/2)
    # For large D, it's approximately N(0, 1/D)
    
    mean_cos = np.mean(cos_sims)
    var_cos = np.var(cos_sims)
    theo_var_cos = 1.0 / d
    
    print(f"Cosine Similarity Distribution (sampled {len(cos_sims)} pairs):")
    print(f"  Observed Mean: {mean_cos:.4f} (Theoretical: 0.0000)")
    print(f"  Observed Var:  {var_cos:.4f} (Theoretical 1/D: {theo_var_cos:.4f})")
    
    # KS Test vs Normal(0, 1/sqrt(D))
    # We allow scaling for cosine similarity too to see how much more correlated the data is than random noise
    obs_std_cos = np.sqrt(var_cos)
    theo_std_cos = np.sqrt(theo_var_cos)
    scale_cos = obs_std_cos / theo_std_cos
    ks_stat_cos, ks_p_cos = stats.kstest(cos_sims, 'norm', args=(mean_cos, obs_std_cos))
    print(f"  KS Test vs N({mean_cos:.4f}, {var_cos:.4f}) (scale {scale_cos:.4f}x theoretical): stat={ks_stat_cos:.4f}, p={ks_p_cos:.4g}")

def main():
    # Tag Vectors
    if os.path.exists("steam_tag_vectors.npy"):
        tag_vectors = np.load("steam_tag_vectors.npy")
        analyze_distribution(tag_vectors, "Steam Tag Vectors")
    
    # Structural Embeddings
    if os.path.exists("embeddings_structural.npy"):
        struct_vectors = np.load("embeddings_structural.npy")
        analyze_distribution(struct_vectors, "Structural Embeddings")

    # Descriptive Embeddings
    if os.path.exists("embeddings_desc.npy"):
        desc_vectors = np.load("embeddings_desc.npy")
        analyze_distribution(desc_vectors, "Descriptive Embeddings")

if __name__ == "__main__":
    main()
