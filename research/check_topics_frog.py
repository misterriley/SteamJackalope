import pandas as pd
import numpy as np
import json
import os
from common.constants import TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR

df = pd.read_parquet('data/production/metadata.parquet', columns=['name', 'appid'])
topic_dist = np.load(TOPIC_DISTRIBUTIONS_FILE, mmap_mode='r')

with open('topic_labels.json', 'r') as f:
    labels = json.load(f)

def print_topics(appid):
    idx = df[df['appid'] == appid].index[0]
    name = df.iloc[idx]['name']
    dist = topic_dist[idx]
    top_indices = np.argsort(dist)[::-1][:5]
    print(f"\n{name} ({appid}) Top Topics:")
    for i in top_indices:
        print(f"  {i}: {dist[i]:.4f} - {labels.get(str(i), 'Unknown')}")

print_topics(1194840) # Frog Fractions
print_topics(1379510) # Algebra Ridge
print_topics(332570)  # Amazing Frog?
