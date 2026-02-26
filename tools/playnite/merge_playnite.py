import json
import pandas as pd
import os

def merge_playnite_conservative(steam_id, playnite_json, master_csv):
    lib_path = f"data/user_{steam_id}_library.csv"
    gt_path = f"data/user_{steam_id}_ground_truth.csv"
    
    # 1. Load Master Metadata
    print("Loading master metadata...")
    master_df = pd.read_csv(master_csv, usecols=['appid', 'name'])
    valid_appids = set(master_df['appid'].unique())
    master_names = dict(zip(master_df['appid'], master_df['name']))
    
    # 2. Load Playnite Data
    with open(playnite_json, 'r') as f:
        playnite_data = json.load(f)
    
    # 3. Load User Data
    if os.path.exists(lib_path):
        lib_df = pd.read_csv(lib_path)
    else:
        lib_df = pd.DataFrame(columns=['appid', 'name', 'playtime_forever', 'user_voted_up', 'user_review_text'])
        
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
    else:
        gt_df = pd.DataFrame(columns=['appid', 'actual_rating', 'ignore', 'status', 'notes'])

    # Ensure 'notes' column exists
    if 'notes' not in gt_df.columns:
        gt_df['notes'] = ""

    # Track existing IDs
    existing_appids = set(lib_df['appid']).union(set(gt_df['appid']))
    
    new_lib_entries = []
    new_gt_entries = []
    
    print(f"Checking {len(playnite_data)} Playnite entries for new games...")
    
    for entry in playnite_data:
        appid = entry['appid']
        
        # Rule: Only add if NOT already in the dataset
        if appid in existing_appids:
            continue
            
        # Rule: Must be a valid Steam game
        if appid not in valid_appids:
            continue
            
        name = master_names[appid]
        playtime_mins = int(entry['playtime_hrs'] * 60)
        
        # Add to Library
        new_lib_entries.append({
            'appid': appid,
            'name': name,
            'playtime_forever': playtime_mins,
            'user_voted_up': False,
            'user_review_text': ""
        })
            
        # Rule: All new games go to backlog, labeled as 'playnite'
        new_gt_entries.append({
            'appid': appid,
            'actual_rating': 5,
            'ignore': False,
            'status': 'backlog',
            'notes': 'playnite'
        })

    # Append new entries
    if new_lib_entries:
        lib_df = pd.concat([lib_df, pd.DataFrame(new_lib_entries)], ignore_index=True)
    if new_gt_entries:
        gt_df = pd.concat([gt_df, pd.DataFrame(new_gt_entries)], ignore_index=True)
        
    # Save
    lib_df.to_csv(lib_path, index=False)
    gt_df.to_csv(gt_path, index=False)
    
    print(f"Conservative Merge Complete for {steam_id}:")
    print(f"- Truly new games added: {len(new_lib_entries)}")
    print(f"- All new entries set to 'backlog' and labeled 'playnite'")
    print(f"Updated {lib_path} and {gt_path}")

if __name__ == "__main__":
    # First, restore from backup again to ensure we have a clean slate for the 'notes' addition
    import shutil
    steam_id = '76561198039155404'
    backup_dir = f"data/backups/user_{steam_id}_pre_playnite"
    for f in os.listdir(backup_dir):
        shutil.copy(os.path.join(backup_dir, f), os.path.join("data", f))
    
    merge_playnite_conservative(steam_id, 'playnite_steam_games.json', 'data/pipeline_games_clean.csv')
