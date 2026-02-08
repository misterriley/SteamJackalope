
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import EMBEDDINGS_DESC_FILE, EMBEDDINGS_TAG_FILE, EPSILON

def precalculate_norm(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    
    print(f"Checking {filepath}...")
    try:
        # Load a small sample to check normalization
        m_sample = np.load(filepath, mmap_mode='r')
        sample_norms = np.linalg.norm(m_sample[:100].astype(np.float32), axis=1)
        if np.allclose(sample_norms, 1.0, atol=1e-3):
            print(f"  -> Already normalized. Skipping.")
            return
        del m_sample # Ensure we release the mmap
    except Exception as e:
        print(f"  -> Check failed ({e}). Proceeding with full normalization.")

    print(f"Loading and normalizing {filepath}...")
    m = np.load(filepath) # Load fully into RAM
    norms = np.linalg.norm(m.astype(np.float32), axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    m_norm = (m / norms).astype(np.float16)
    
    temp_path = filepath + ".tmp.npy"
    np.save(temp_path, m_norm)
    
    # Atomic replace (on POSIX) or Replace (on Windows)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(temp_path, filepath)
    
    print(f"Saved normalized {filepath} (shape: {m_norm.shape}, dtype: {m_norm.dtype})")

if __name__ == "__main__":
    precalculate_norm(EMBEDDINGS_DESC_FILE)
    precalculate_norm(EMBEDDINGS_TAG_FILE)
