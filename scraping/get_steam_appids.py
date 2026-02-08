import requests
import pandas as pd
import time
import os
import sys
from tqdm import tqdm

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import API_KEY

def fetch_and_save_appids(output_file="data/steam_appids.csv"):
    """
    Fetch the full list of AppIDs from the Steam API using pagination
    and save them to a CSV file.
    """
    print("Fetching full AppID list from Steam...")
    all_apps = []
    last_appid = 0
    have_more_results = True
    
    pbar = tqdm(desc="Fetching AppIDs", unit=" apps", unit_scale=True)
    while have_more_results:
        # Fallback to public endpoint if API_KEY is not set or invalid
        if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
            url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
            is_public = True
        else:
            is_public = False
            url = f"https://api.steampowered.com/IStoreService/GetAppList/v1/?key={API_KEY}&include_games_only=true&max_results=10000&last_appid={last_appid}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                json_data = response.json()
                if is_public:
                    data = json_data.get('applist', {})
                else:
                    data = json_data.get('response', {})
                
                apps = data.get('apps', [])
                
                # Each app dict contains: appid, name, last_modified, price_change_number
                all_apps.extend(apps)
                
                if is_public:
                    have_more_results = False
                else:
                    have_more_results = data.get('have_more_results', False)
                    last_appid = data.get('last_appid', 0)
                
                pbar.update(len(apps))
                
                # Small sleep to respect API limits
                time.sleep(0.2)
            else:
                print(f"Failed to fetch: Status Code {response.status_code}")
                if response.status_code == 403 and API_KEY and API_KEY != "YOUR_API_KEY_HERE":
                    print("Received 403 Forbidden. Your API Key might be invalid. Attempting fallback to public endpoint...")
                    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json().get('applist', {})
                        apps = data.get('apps', [])
                        all_apps.extend(apps)
                        print(f"Successfully fetched {len(apps)} apps from public endpoint.")
                        have_more_results = False # Public endpoint isn't paginated the same way
                        pbar.update(len(apps))
                        break

                break
        except Exception as e:
            print(f"Error during fetch: {e}")
            break
    
    pbar.close()
            
    if all_apps:
        df = pd.DataFrame(all_apps)
        # Reorder columns for convenience
        cols = ['appid', 'name', 'last_modified', 'price_change_number']
        df = df[[c for c in cols if c in df.columns]]
        
        df.to_csv(output_file, index=False)
        print(f"\nSuccess! Saved {len(all_apps)} AppIDs to {output_file}")
    else:
        print("No apps were retrieved.")

if __name__ == "__main__":
    fetch_and_save_appids()
