
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR

def generate_topic_stats():
    print("Calculating per-topic standardization stats...")
    data = np.load(TOPIC_DISTRIBUTIONS_FILE).astype(np.float32)
    
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Save as .npy files for the solver to load
    np.save(os.path.join(PRODUCTION_DATA_DIR, "topic_means.npy"), means)
    np.save(os.path.join(PRODUCTION_DATA_DIR, "topic_stds.npy"), stds)
    
    print(f"Saved stats for {len(means)} topics.")

if __name__ == "__main__":
    generate_topic_stats()
