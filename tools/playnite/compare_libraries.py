import json
import pandas as pd
import os

def find_truly_missing_games(playnite_json, steam_id):
    lib_path = f"data/user_{steam_id}_library.csv"
    gt_path = f"data/user_{steam_id}_ground_truth.csv"
    
    if not os.path.exists(playnite_json):
        print("Playnite JSON not found.")
        return

    # Load Playnite data
    with open(playnite_json, 'r') as f:
        playnite_games = json.load(f)
    
    # Load User Library
    known_appids = set()
    if os.path.exists(lib_path):
        lib_df = pd.read_csv(lib_path)
        known_appids.update(lib_df['appid'].unique())
        print(f"Loaded {len(known_appids)} from library.csv")
        
    # Load Ground Truth
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        gt_ids = set(gt_df['appid'].unique())
        print(f"Loaded {len(gt_ids)} from ground_truth.csv")
        known_appids.update(gt_ids)
    
    print(f"Total unique known games in system: {len(known_appids)}")
    
    missing_games = []
    for g in playnite_games:
        if g['appid'] not in known_appids:
            missing_games.append(g)
            
    print(f"\nFound {len(missing_games)} games in Playnite that are NOT in library or ground_truth:")
    
    # Sort by playtime
    missing_games = sorted(missing_games, key=lambda x: x['playtime_hrs'], reverse=True)
    
    for g in missing_games[:50]:
        print(f"- {g['name']} ({g['appid']}): {g['playtime_hrs']} hrs")
        
    if len(missing_games) > 50:
        print(f"... and {len(missing_games) - 50} more.")

if __name__ == "__main__":
    find_truly_missing_games('playnite_steam_games.json', '76561198039155404')
