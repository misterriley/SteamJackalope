import json
import pandas as pd
import os

def enrich_playnite_names(json_path, csv_path):
    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        return
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    # Load Playnite data
    with open(json_path, 'r') as f:
        playnite_games = json.load(f)

    # Load Steam metadata mapping
    # We only need appid and name
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['appid', 'name'])
    
    # Create a mapping dictionary
    mapping = dict(zip(df['appid'], df['name']))
    print(f"Loaded {len(mapping)} mappings.")

    # Enrich names
    updated_count = 0
    for game in playnite_games:
        appid = game['appid']
        if appid in mapping:
            original_name = game.get('name', 'Unknown')
            steam_name = mapping[appid]
            
            # Update if it was Unknown, or if we have a "messy" name
            # Messy names usually have tags, or are very long, or are just "Steam"/"Official"
            if original_name == "Unknown" or len(original_name) > 50 or "<" in original_name or ":" in original_name:
                game['name'] = steam_name
                updated_count += 1
            elif original_name.lower() in ["steam", "official", "wikipedia", "wikia"]:
                game['name'] = steam_name
                updated_count += 1

    # Save updated JSON
    with open(json_path, 'w') as f:
        json.dump(playnite_games, f, indent=4)
    
    print(f"Updated {updated_count} game names in {json_path}")

if __name__ == "__main__":
    enrich_playnite_names('playnite_steam_games.json', 'data/pipeline_games_clean.csv')
