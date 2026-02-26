import pandas as pd
import os

def check_status_changes(steam_id, backup_dir):
    current_gt_path = f"data/user_{steam_id}_ground_truth.csv"
    backup_gt_path = os.path.join(backup_dir, f"user_{steam_id}_ground_truth.csv")
    master_csv = "data/pipeline_games_clean.csv"

    if not os.path.exists(current_gt_path) or not os.path.exists(backup_gt_path):
        print("Missing current or backup ground truth file.")
        return

    current_df = pd.read_csv(current_gt_path)
    backup_df = pd.read_csv(backup_gt_path)
    master_df = pd.read_csv(master_csv, usecols=['appid', 'name'])
    names = dict(zip(master_df['appid'], master_df['name']))

    merged = current_df.merge(backup_df, on='appid', suffixes=('_curr', '_back'), how='left')
    
    changed = merged[
        (merged['status_curr'] == 'played') & 
        (merged['status_back'] != 'played') & 
        (merged['status_back'].notna())
    ]
    
    new_played = merged[
        (merged['status_curr'] == 'played') & 
        (merged['status_back'].isna())
    ]

    print("Games updated from another status to 'played' (" + str(len(changed)) + "):")
    for _, row in changed.head(20).iterrows():
        name = names.get(row['appid'], "Unknown")
        print("- " + str(name) + " (" + str(row['appid']) + "): " + str(row['status_back']) + " -> played")
    
    if len(changed) > 20:
        print("... and " + str(len(changed) - 20) + " more updates.")

    print("\nNew games added as 'played' from Playnite (" + str(len(new_played)) + "):")
    for _, row in new_played.head(20).iterrows():
        name = names.get(row['appid'], "Unknown")
        print("- " + str(name) + " (" + str(row['appid']) + ")")
    
    if len(new_played) > 20:
        print("... and " + str(len(new_played) - 20) + " more additions.")

if __name__ == "__main__":
    check_status_changes('76561198039155404', 'data/backups/user_76561198039155404_pre_playnite')
