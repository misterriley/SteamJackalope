import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import TOPIC_DISTRIBUTIONS_FILE

def check_topic_variance():
    if not os.path.exists(TOPIC_DISTRIBUTIONS_FILE):
        print("Error: Topic distributions file not found.")
        return

    print("Loading topic distributions...")
    data = np.load(TOPIC_DISTRIBUTIONS_FILE).astype(np.float32)
    
    variances = np.var(data, axis=0)
    means = np.mean(data, axis=0)
    
    print("\n--- Topic Variance Statistics ---")
    print("Count:    " + str(len(variances)))
    print("Mean Var: " + f"{np.mean(variances):.8f}")
    print("Std Var:  " + f"{np.std(variances):.8f}")
    print("Min Var:  " + f"{np.min(variances):.8f}")
    print("Max Var:  " + f"{np.max(variances):.8f}")
    
    ratio = np.max(variances)/np.min(variances) if np.min(variances) > 0 else float('inf')
    print("Ratio (Max/Min): " + f"{ratio:.2f}")

    top_5_idx = np.argsort(-variances)[:5]
    print("\n--- Top 5 High-Variance Topics ---")
    for idx in top_5_idx:
        print("Topic " + str(idx) + ": Var=" + f"{variances[idx]:.6f}" + ", Mean=" + f"{means[idx]:.6f}")

    bot_5_idx = np.argsort(variances)[:5]
    print("\n--- Bottom 5 Low-Variance Topics ---")
    for idx in bot_5_idx:
        print("Topic " + str(idx) + ": Var=" + f"{variances[idx]:.6f}" + ", Mean=" + f"{means[idx]:.6f}")

if __name__ == "__main__":
    check_topic_variance()
