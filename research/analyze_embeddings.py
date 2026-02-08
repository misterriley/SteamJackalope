import numpy as np

def analyze_embeddings(name, file_path):
    print(f"--- Analyzing {name} ({file_path}) ---")
    try:
        embeddings = np.load(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return

    print(f"Shape: {embeddings.shape}")
    
    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    
    print(f"Mean (first 5 elements): {mean[:5]}")
    print(f"Mean min/max: {np.min(mean):.6f}, {np.max(mean):.6f}")
    print(f"Mean abs average: {np.mean(np.abs(mean)):.6f}")
    
    diag = np.diag(cov)
    zero_var_dims = np.where(diag < 1e-5)[0]
    print(f"Covariance diagonal (first 5): {diag[:5]}")
    print(f"Covariance diagonal sorted smallest 10: {np.sort(diag)[:10]}")
    print(f"Covariance diagonal min/max: {np.min(diag):.6f}, {np.max(diag):.6f}")
    print(f"Covariance diagonal average: {np.mean(diag):.6f}")
    print(f"Number of zero-variance dimensions (threshold 1e-5): {len(zero_var_dims)}")
    if len(zero_var_dims) > 0:
        print(f"Zero-variance indices: {zero_var_dims}")
    
    # Off-diagonal elements
    off_diag = cov - np.diag(diag)

    print(f"Off-diagonal min/max: {np.min(off_diag):.6f}, {np.max(off_diag):.6f}")
    print(f"Off-diagonal abs average: {np.mean(np.abs(off_diag)):.6f}")
    
    identity = np.eye(cov.shape[0])
    dist_to_identity = np.linalg.norm(cov - identity)
    print(f"Frobenius distance to identity: {dist_to_identity:.6f}")
    print("\n")

if __name__ == "__main__":
    analyze_embeddings("Structural", "embeddings_structural.npy")
    analyze_embeddings("Descriptive", "embeddings_desc.npy")
