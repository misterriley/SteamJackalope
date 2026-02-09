"""
Test script to evaluate variance-based dimensionality reduction for whitening matrices.
"""
import numpy as np

def analyze_whitening_matrix(filepath, name):
    """Analyze a whitening matrix to determine dimensions needed for 80% variance."""
    W = np.load(filepath).astype(np.float32)
    # The whitening matrix W transforms centered vectors: whitened = centered @ W
    # The covariance of whitened vectors should be identity: W^T @ M_centered @ W = I
    # To find effective dimensionality, we can look at singular values of W or the covariance
    
    # Since W is a square transformation matrix, we can compute the effective dimensionalities
    # by looking at the eigenvalues of the covariance in the whitened space.
    # Actually simpler: The number of non-zero dimensions is determined by the rank of W.
    
    # We can compute the proportion of variance explained by each dimension by looking at
    # the diagonal of the covariance in the original space after whitening.
    # But we want to know: if we truncate W to fewer columns, what % of variance is retained?
    
    # Approach: Compute U, S, Vt of W. The singular values of W tell us how much variance each
    # whitened dimension explains when transformed back to the centered space.
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    cumvar = np.cumsum(S) / np.sum(S)
    
    print(f"\n{name}:")
    print(f"  Full shape: {W.shape}")
    print(f"  Singular values (first 10): {S[:10]}")
    print(f"  Dimensions for 80% variance: {np.argmax(cumvar >= 0.8) + 1}")
    print(f"  Dimensions for 90% variance: {np.argmax(cumvar >= 0.9) + 1}")
    print(f"  Dimensions for 95% variance: {np.argmax(cumvar >= 0.95) + 1}")
    
    return cumvar

def analyze_embeddings_and_whitening(embeddings_path, W_path, mean_path, name):
    """Analyze embeddings and corresponding whitening matrix together."""
    print(f"\n=== Analyzing {name} ===")
    
    # Load embeddings (float16) and convert to float32 for SVD
    embeddings = np.load(embeddings_path).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Load whitening matrix and mean
    W = np.load(W_path).astype(np.float32)
    mean = np.load(mean_path).astype(np.float32)
    
    # Center embeddings
    centered = embeddings - mean
    
    # Apply whitening: whitened = centered @ W
    whitened = centered @ W
    
    # Compute covariance of whitened embeddings
    cov_whitened = np.cov(whitened, rowvar=False)
    U, S, Vt = np.linalg.svd(cov_whitened)
    cumvar = np.cumsum(S) / np.sum(S)
    
    print(f"Whitening matrix shape: {W.shape}")
    print(f"Whitened embeddings covariance singular values (first 10): {S[:10]}")
    print(f"Dimensions for 80% variance in whitened space: {np.argmax(cumvar >= 0.8) + 1}")
    print(f"Dimensions for 90% variance: {np.argmax(cumvar >= 0.9) + 1}")
    print(f"Dimensions for 95% variance: {np.argmax(cumvar >= 0.95) + 1}")
    
    # Also analyze W directly
    analyze_whitening_matrix(W_path, f"W matrix for {name}")
    
    return whitened.shape[1]

if __name__ == "__main__":
    # Analyze semantic embeddings
    desc_dims = analyze_embeddings_and_whitening(
        'embeddings_desc.npy',
        'w_desc.npy',
        'mean_desc.npy',
        'Descriptive Embeddings'
    )
    
    structural_dims = analyze_embeddings_and_whitening(
        'embeddings_structural.npy',
        'w_structural.npy',
        'mean_structural.npy',
        'Structural Embeddings'
    )
    
    # Analyze tag vectors separately
    print(f"\n=== Analyzing Tag Vectors ===")
    tag_vectors = np.load('steam_tag_vectors.npy').astype(np.float32)
    print(f"Tag vectors shape: {tag_vectors.shape}")
    cov_tags = np.cov(tag_vectors, rowvar=False)
    U, S, Vt = np.linalg.svd(cov_tags)
    cumvar = np.cumsum(S) / np.sum(S)
    print(f"Tag vector covariance singular values (first 10): {S[:10]}")
    print(f"Dimensions for 80% variance: {np.argmax(cumvar >= 0.8) + 1}")
    print(f"Dimensions for 90% variance: {np.argmax(cumvar >= 0.9) + 1}")
    print(f"Dimensions for 95% variance: {np.argmax(cumvar >= 0.95) + 1}")
    
    # Check w_tag
    w_tag = np.load('w_tag.npy')
    print(f"\nw_tag.npy shape: {w_tag.shape}, values: {w_tag}")