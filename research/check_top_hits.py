import pandas as pd
import numpy as np
import os
import json

def get_top_recommendations():
    sid = '76561198039155404'
    gt_path = f'data/user_{sid}_ground_truth.csv'
    ratings_path = f'data/user_{sid}_predicted_ratings.npy'
    metadata_path = 'data/production/metadata.parquet'
    
    # 1. Load Data
    df_gt = pd.read_csv(gt_path)
    full_metadata = pd.read_parquet(metadata_path)
    scores = np.load(ratings_path)
    
    # Map appid to index
    appid_to_idx = {int(aid): idx for idx, aid in enumerate(full_metadata['appid'])}
    
    # 2. Top Backlog
    backlog_appids = df_gt[df_gt['status'] == 'backlog']['appid'].tolist()
    backlog_data = []
    for aid in backlog_appids:
        if aid in appid_to_idx:
            idx = appid_to_idx[aid]
            name = full_metadata.iloc[idx]['name']
            score = scores[idx]
            backlog_data.append({'appid': aid, 'name': name, 'score': score})
            
    top_backlog = sorted(backlog_data, key=lambda x: x['score'], reverse=True)[:10]
    
    # 3. Top Discovery (Out-of-Database / Not in Library)
    # Statuses that count as "In Library"
    in_library_appids = set(df_gt[df_gt['status'].isin(['rated', 'played', 'backlog'])]['appid'].tolist())
    
    discovery_data = []
    # Filter for games NOT in library and NOT ignored
    ignored_appids = set(df_gt[df_gt['status'] == 'ignored']['appid'].tolist())
    exclude = in_library_appids.union(ignored_appids)
    
    # We only want to look at a subset of high-quality/popular games for speed or just use the pre-filtered top_recommendations
    # Let's use the full scores but filter out library
    for i in range(len(full_metadata)):
        aid = int(full_metadata.iloc[i]['appid'])
        if aid not in exclude:
            discovery_data.append({'appid': aid, 'name': full_metadata.iloc[i]['name'], 'score': scores[i]})
            
    top_discovery = sorted(discovery_data, key=lambda x: x['score'], reverse=True)[:10]

    print("\n=== TOP BACKLOG GAMES (Core 9) ===")
    for i, g in enumerate(top_backlog):
        print(f"{i+1}. {g['name']} ({g['appid']}): {g['score']:.2f}")
        
    print("\n=== TOP DISCOVERY GAMES (Out-of-Library) ===")
    for i, g in enumerate(top_discovery):
        print(f"{i+1}. {g['name']} ({g['appid']}): {g['score']:.2f}")

if __name__ == '__main__':
    get_top_recommendations()
