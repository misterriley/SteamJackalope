import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import TAG_VECTORS_FILE, TAG_NORMS_FILE

def main():
    if not os.path.exists(TAG_VECTORS_FILE):
        print(f"Error: {TAG_VECTORS_FILE} not found.")
        return

    print(f"Loading {TAG_VECTORS_FILE}...")
    tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
    print(f"Tag vectors shape: {tag_vectors.shape}")

    print("Calculating norms...")
    # Calculate in chunks to avoid RAM spike if necessary, 
    # but 155k * 455 float16 is ~140MB, so it should be fine to do at once if we convert to float32.
    # Actually, let's do it in chunks just to be safe and memory efficient.
    
    num_games = tag_vectors.shape[0]
    norms = np.zeros(num_games, dtype=np.float16)
    chunk_size = 10000
    
    for i in range(0, num_games, chunk_size):
        end = min(i + chunk_size, num_games)
        chunk = tag_vectors[i:end].astype(np.float32)
        norms[i:end] = np.linalg.norm(chunk, axis=1).astype(np.float16)
        if end % 50000 == 0 or end == num_games:
            print(f"  Processed {end}/{num_games}...")

    print(f"Saving norms to {TAG_NORMS_FILE}...")
    np.save(TAG_NORMS_FILE, norms)
    print("Done.")

if __name__ == "__main__":
    main()
