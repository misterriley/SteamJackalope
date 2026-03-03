import json
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    W_DESC_FILE, MEAN_DESC_FILE, MODEL_NAME, TAG_NAMES_FILE
)

def describe():
    motivations = ["Destruction", "Excitement", "Competition", "Community", "Challenge", "Strategy", "Completion", "Power", "Fantasy", "Story", "Discovery", "Design"]
    
    W = np.load(W_DESC_FILE)
    mean = np.load(MEAN_DESC_FILE)
    model = SentenceTransformer(MODEL_NAME)
    mot_raw = model.encode(motivations).astype(np.float32)
    mot_whitened = (mot_raw - mean) @ W
    
    with open("research/tag_semantic_correlations.json", "r") as f:
        corrs = json.load(f)
        
    with open(TAG_NAMES_FILE, 'r') as f:
        tag_names = json.load(f)
            
    num_tags = len(tag_names)
    num_dims = 235
    R = np.zeros((num_tags, num_dims))
    t_to_i = {t: i for i, t in enumerate(tag_names)}
    
    for d_str, data in corrs.items():
        d = int(d_str)
        if d >= num_dims: continue
        for t, s in data['top_positive']:
            if t in t_to_i: R[t_to_i[t], d] = s
        for t, s in data['top_negative']:
            if t in t_to_i: R[t_to_i[t], d] = s 
                
    for m_idx, m_name in enumerate(motivations):
        print("\n" + "="*80)
        print("MOTIVATION: " + m_name)
        m_vec = mot_whitened[m_idx]
        tag_scores = R @ m_vec
        
        top_tags_idx = np.argsort(-tag_scores)[:10]
        bottom_tags_idx = np.argsort(tag_scores)[:10]
        
        print("Strongest Correlating Tags:")
        for idx in top_tags_idx:
            if tag_scores[idx] > 0.05:
                print("  (+) " + tag_names[idx] + " (" + str(round(float(tag_scores[idx]), 3)) + ")")
        print("Opposing Tags:")
        for idx in bottom_tags_idx:
            if tag_scores[idx] < -0.05:
                print("  (-) " + tag_names[idx] + " (" + str(round(float(tag_scores[idx]), 3)) + ")")

        top_dims = np.argsort(-np.abs(m_vec))[:3]
        print("Key Semantic Components:")
        for d in top_dims:
            weight = m_vec[d]
            sign = "(+)" if weight > 0 else "(-)"
            dir_data = corrs[str(d)]
            dir_tags = dir_data['top_positive' if weight > 0 else 'top_negative']
            tag_str = ", ".join([t for t, s in dir_tags[:3]])
            print("  " + sign + " Dim " + str(d) + " [" + tag_str + "] (Weight: " + str(round(float(weight), 2)) + ")")

if __name__ == "__main__":
    describe()
