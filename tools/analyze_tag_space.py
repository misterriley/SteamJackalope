import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import TAG_VECTORS_FILE, W_TAG_FILE

def analyze_tag_space():
    if not os.path.exists(W_TAG_FILE):
        print(f"Whitening matrix not found at {W_TAG_FILE}")
        return

    W = np.load(W_TAG_FILE)
    print(f"Whitening matrix shape: {W.shape}")
    
    # If W is (OriginalTags, WhitenedDim)
    # The WhitenedDim is the number of components kept.
    whitened_dim = W.shape[1]
    
    # We can't see the singular values directly from W because it's U * diag(1/sqrt(S))
    # But we can look at the taste profile vibe vector.
    
    # Let's try to find the transformed vectors if they exist, or reconstruct them.
    # Actually, let's just look at the code and see if we can find where S comes from.
    
    print(f"Current whitened dimension: {whitened_dim}")
    
    # If the user wants to increase variance to 95%, and it's currently at 0.95 in the code,
    # maybe the artifacts were generated with an OLD version of the code (e.g. 80%).
    
if __name__ == "__main__":
    analyze_tag_space()
