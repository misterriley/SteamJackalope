
import requests
import json
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_trending_games(output_file="data/trending_appids.json"):
    """
    Fetch the top 100 most played games from Steam Charts API
    and save them to a JSON file.
    """
    print("Fetching top 100 most played games from Steam...")
    url = "https://api.steampowered.com/ISteamChartsService/GetMostPlayedGames/v1/"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            ranks = json_data.get('response', {}).get('ranks', [])
            
            if not ranks:
                print("No ranks found in API response.")
                return
            
            # Extract appids
            appids = [rank['appid'] for rank in ranks]
            
            # Ensure the data directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(appids, f)
            
            print(f"Success! Saved {len(appids)} trending AppIDs to {output_file}")
        else:
            print(f"Failed to fetch: Status Code {response.status_code}")
    except Exception as e:
        print(f"Error during fetch: {e}")

if __name__ == "__main__":
    fetch_trending_games()
