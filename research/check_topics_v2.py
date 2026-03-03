import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.constants import *

def check_topics_v2():
    df = pd.read_parquet(METADATA_FILE)
    aid_to_idx = {int(a): i for i, a in enumerate(df['appid'])}
    
    apps = {
        "Detroit": 1222140,
        "Suck Up!": 2726370,
        "A Way Out": 1222700,
        "Ourea": 1250030
    }
    
    top_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')
    
    for name, aid in apps.items():
        if aid in aid_to_idx:
            idx = aid_to_idx[aid]
            dist = top_dist[idx]
            top_3 = np.argsort(-dist)[:3]
            print(f"{name} ({aid}):")
            for t in top_3:
                print(f"  Topic {t}: {dist[t]:.4f}")

if __name__ == '__main__':
    check_topics_v2()
