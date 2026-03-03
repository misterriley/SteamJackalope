import numpy as np
import json
import scipy.linalg
from sklearn.model_selection import KFold

# Load verbs
with open("tag_categories.json", "r") as f:
    verbs = sorted(json.load(f)["verbs"])

# Load Jaccard
A = np.load("data/production/verb_jaccard_matrix.npy")
# Remove diagonal for adjacency
np.fill_diagonal(A, 0)

# Prepare for CV
# Get all upper triangular indices
i_upper, j_upper = np.triu_indices(A.shape[0], k=1)
edges = np.column_stack((i_upper, j_upper))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

t_values = np.logspace(-1, 1.5, 30)  # ~0.1 to ~31.6
correlations = {t: [] for t in t_values}

print("Running 5-fold CV to find optimal diffusion time t...")

for fold, (train_idx, test_idx) in enumerate(kf.split(edges)):
    A_train = np.copy(A)
    # Mask out test edges
    test_edges = edges[test_idx]
    for i, j in test_edges:
        A_train[i, j] = 0
        A_train[j, i] = 0
        
    # Compute Normalized Laplacian
    # Add a small epsilon to degree to avoid division by zero if isolated
    D = np.sum(A_train, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-9)
    D_inv_sqrt[D == 0] = 0
    L = np.eye(A.shape[0]) - D_inv_sqrt[:, None] * A_train * D_inv_sqrt[None, :]
    
    # Precompute eigendecomposition for faster expm
    evals, evecs = np.linalg.eigh(L)
    
    true_vals = A[test_edges[:, 0], test_edges[:, 1]]
    
    for t in t_values:
        # exp(-tL) = Q exp(-t Lambda) Q^T
        exp_evals = np.exp(-t * evals)
        H_t = evecs @ np.diag(exp_evals) @ evecs.T
        
        pred_vals = H_t[test_edges[:, 0], test_edges[:, 1]]
        # We can just correlate predicted with true
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
        correlations[t].append(corr)

mean_corrs = {t: np.mean(corrs) for t, corrs in correlations.items()}

best_t = max(mean_corrs, key=mean_corrs.get)
print(f"Optimal t: {best_t:.4f} with mean correlation: {mean_corrs[best_t]:.4f}")

# Now compute final diffusion matrix with full A and best_t
best_t = 2.0
D = np.sum(A, axis=1)
D_inv_sqrt = 1.0 / np.sqrt(D + 1e-9)
D_inv_sqrt[D == 0] = 0
L = np.eye(A.shape[0]) - D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
evals, evecs = np.linalg.eigh(L)
exp_evals = np.exp(-best_t * evals)
H_final = evecs @ np.diag(exp_evals) @ evecs.T

# Normalize H_final to have 1s on the diagonal (Cosine similarity in diffusion space)
diag = np.diag(H_final)
H_norm = H_final / np.sqrt(diag[:, None] * diag[None, :])

np.save("data/production/verb_diffusion_matrix.npy", H_norm)

# Save JSON for inspection
results = {}
for i, v1 in enumerate(verbs):
    results[v1] = {}
    for j, v2 in enumerate(verbs):
        if i != j:
            results[v1][v2] = round(float(H_norm[i, j]), 4)

with open("verb_diffusion_similarities.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved data/production/verb_diffusion_matrix.npy and verb_diffusion_similarities.json")
