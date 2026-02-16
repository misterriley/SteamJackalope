import pandas as pd
import numpy as np
from scipy.stats import norm
import sys
import os

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import METADATA_FILE, QUALITY_SCORE_S_CONST, GLOBAL_POSITIVE_RATE

def find_anchors():
    df = pd.read_parquet(METADATA_FILE, columns=['appid', 'name', 'positive', 'negative'])
    p = df['positive'].fillna(0).values
    n = df['negative'].fillna(0).values
    s = QUALITY_SCORE_S_CONST
    a = GLOBAL_POSITIVE_RATE
    
    prob = (p + s * a) / (p + n + s)
    q = norm.ppf(np.clip(prob, 1e-6, 1-1e-6))
    
    idx_max = np.argmax(q)
    idx_min = np.argmin(q)
    
    q_max = q[idx_max]
    q_min = q[idx_min]
    
    # Calculate Hard Labels for the absolute extremes
    # p+ = 1 for the best game
    q_pers_max = q_max + norm.pdf(q_max) / norm.cdf(q_max)
    # p+ = 0 for the worst game
    q_pers_min = q_min - norm.pdf(q_min) / norm.sf(q_min)
    
    print(f"Anchors identified at s={s}:")
    print(f"MAX: {df.iloc[idx_max]['name']} (Q={q_max:.4f}, Q_pers_max={q_pers_max:.4f})")
    print(f"MIN: {df.iloc[idx_min]['name']} (Q={q_min:.4f}, Q_pers_min={q_pers_min:.4f})")
    
    # Calculate m and c for: Rating = m * Q_pers + c
    # 10 = m * q_pers_max + c
    # 0 = m * q_pers_min + c
    m = 10 / (q_pers_max - q_pers_min)
    c = -m * q_pers_min
    
    print("\nDerived mapping coefficients:")
    print(f"m: {m:.6f}")
    print(f"c: {c:.6f}")
    print(f"Formula: Rating = {m:.4f} * Q_pers + {c:.4f}")

if __name__ == "__main__":
    find_anchors()
