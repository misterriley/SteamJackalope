import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import *

def test_prompt(prompt_text):
    print(f"Testing Prompt: '{prompt_text}'")
    model = SentenceTransformer(MODEL_NAME)
    prompt_vec = model.encode([prompt_text])[0]
    
    w_desc = np.load(W_DESC_FILE)
    emb_desc = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
    
    p_whitened = np.dot(prompt_vec, w_desc)
    p_norm = p_whitened / np.linalg.norm(p_whitened)
    
    dots = np.dot(emb_desc.astype(np.float32), p_norm)
    
    print(f"Raw Dots - Max: {np.max(dots):.4f}, Mean: {np.mean(dots):.4f}, Std: {np.std(dots):.4f}")
    
    # Check if we have constants
    mean = 0.0
    std = 1.0
    if os.path.exists(REGULARIZATION_FILE):
        import json
        reg = json.load(open(REGULARIZATION_FILE))
        mean = reg.get('SEMANTIC_SIMILARITY_MEAN', 0.0)
        std = reg.get('SEMANTIC_SIMILARITY_STD', 1.0)
        print(f"Using Constants - Mean: {mean}, Std: {std}")
    
    z_scores = (dots - mean) / std
    print(f"Z-Scores - Max: {np.max(z_scores):.4f}, Min: {np.min(z_scores):.4f}")

if __name__ == "__main__":
    test_prompt("a relaxing farming simulator")
