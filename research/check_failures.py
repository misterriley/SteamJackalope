import pandas as pd
import numpy as np
import json

df = pd.read_parquet('data/production/metadata.parquet', columns=['appid', 'name', 'tags'])

with open('data/production/topic_descriptions.json', 'r') as f:
    topic_desc = json.load(f)

def get_profile(name):
    try:
        idx = df[df['name'] == name].index[0]
        dist = np.load('data/production/topic_distributions.npy', mmap_mode='r')[idx]
        t_means = np.load('data/production/topic_means.npy')
        t_stds = np.load('data/production/topic_stds.npy')
        z = (dist - t_means) / (t_stds + 1e-9)
        top_t = np.argsort(-z)[:5]
        topics = []
        for t in top_t:
            topics.append({"id": int(t), "z": float(z[t]), "desc": topic_desc.get(str(t), f"Topic {t}")})
        return {"name": name, "tags": df.iloc[idx]['tags'], "topics": topics}
    except:
        return f"Error: {name} not found."

games = ['Antichamber', 'Super reaKtor', 'Super Dungeon Boy', 'Protect Your Fool', 'Orgasm Lab Simulator 💦🍌', 'Not Sure About That', 'Slender Threads']
results = [get_profile(n) for n in games]

print(json.dumps(results, indent=2))
