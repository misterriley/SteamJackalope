import requests
import pandas as pd
import numpy as np
import os
import sys
import json
import re

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import API_KEY

def resolve_vanity_url(vanity_url):
    """
    Resolves a Steam vanity URL (e.g., 'mister_jackalope') to a 64-bit SteamID.
    """
    if not API_KEY:
        print("Error: STEAM_API_KEY not found in environment or constants.")
        return None

    url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/?key={API_KEY}&vanityurl={vanity_url}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get('response', {})
            if data.get('success') == 1:
                return data.get('steamid')
            else:
                print(f"Failed to resolve vanity URL: {data.get('message', 'Unknown error')}")
        else:
            print(f"Error: Steam API returned status code {response.status_code}")
    except Exception as e:
        print(f"Exception during vanity URL resolution: {e}")
    
    return None

def fetch_user_library(steamid):
    """
    Fetches the list of owned games and their playtimes for a given SteamID.
    """
    if not API_KEY:
        print("Error: STEAM_API_KEY not found in environment or constants.")
        return None

    # include_appinfo=1 returns game names and icons
    # include_played_free_games=1 includes f2p games in the library
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={API_KEY}&steamid={steamid}&include_appinfo=1&include_played_free_games=1&format=json"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get('response', {})
            games = data.get('games', [])
            if not games:
                print(f"No games found for SteamID {steamid}. The profile might be private.")
                return None
            
            df = pd.DataFrame(games)
            # Reorder and filter columns
            cols = ['appid', 'name', 'playtime_forever', 'playtime_2weeks']
            df = df[[c for c in cols if c in df.columns]]
            return df
        else:
            print(f"Error: Steam API returned status code {response.status_code}")
    except Exception as e:
        print(f"Exception during library fetch: {e}")
    
    return None

import re

from bs4 import BeautifulSoup

def fetch_user_reviews(steamid):
    """
    Scrapes a user's reviews from their Steam Community profile.
    """
    url = f"https://steamcommunity.com/profiles/{steamid}/recommended/"
    
    try:
        response = requests.get(url, cookies={'birthtime': '283996801', 'lastagecheckage': '1-0-1979'})
        if response.status_code != 200:
            print(f"Failed to fetch reviews page: {response.status_code}")
            return pd.DataFrame(columns=['appid', 'user_voted_up'])
            
        soup = BeautifulSoup(response.content, 'html.parser')
        return fetch_user_reviews_from_soup(soup)
    except Exception as e:
        print(f"Exception during review fetch: {e}")
    
    return pd.DataFrame(columns=['appid', 'user_voted_up'])

def fetch_user_reviews_from_soup(soup):
    """
    Parses reviews from a Steam recommended page soup.
    """
    try:
        # In the /recommended/ view, each review is often in a div with class 'review_box'
        review_cards = soup.find_all('div', class_='review_box')
        
        if not review_cards:
            # Try to find elements that look like review links
            review_cards = soup.find_all('div', id=re.compile(r'^review_\d+'))
            
        if not review_cards:
            # Check if there are any links that look like /recommended/
            links = soup.find_all('a', href=re.compile(r'/recommended/\d+'))
            if links:
                reviews = []
                for link in links:
                    appid_match = re.search(r'/recommended/(\d+)', link['href'])
                    if not appid_match: continue
                    appid = int(appid_match.group(1))
                    voted_up = "Recommended" in link.get_text()
                    reviews.append({'appid': appid, 'user_voted_up': voted_up})
                return pd.DataFrame(reviews).drop_duplicates()

            return pd.DataFrame(columns=['appid', 'user_voted_up'])
            
        reviews = []
        for card in review_cards:
            links = card.find_all('a', href=True)
            appid = None
            for link in links:
                match = re.search(r'/(app|recommended)/(\d+)', link['href'])
                if match:
                    appid = int(match.group(2))
                    break
            
            if appid is None: continue
            
            text_div = card.find('div', class_='review_info_content')
            review_text = text_div.get_text(strip=True) if text_div else ""
            
            voted_up = None
            vote_img = card.find('img', src=re.compile(r'icon_thumbs(Up|Down)\.png'))
            if vote_img:
                voted_up = "Up" in vote_img['src']
            
            if voted_up is None:
                header = card.find('div', class_='reviewInfo')
                if header:
                    header_text = header.get_text()
                    if "Not Recommended" in header_text:
                        voted_up = False
                    elif "Recommended" in header_text:
                        voted_up = True
            
            if voted_up is None:
                card_text = card.get_text()
                if "Not Recommended" in card_text:
                    voted_up = False
                elif "Recommended" in card_text:
                    voted_up = True
            
            if voted_up is not None:
                reviews.append({
                    'appid': appid,
                    'user_voted_up': voted_up,
                    'user_review_text': review_text
                })
            
        return pd.DataFrame(reviews).drop_duplicates()
    except Exception as e:
        print(f"Error parsing soup: {e}")
    return pd.DataFrame(columns=['appid', 'user_voted_up'])

def parse_steam_library_html(html_content):
    """
    Parses a user's library from HTML content (useful for private profiles 
    where the user can copy-paste their library page source).
    
    Looks for the 'rgGames' variable in the script tags.
    """
    # Pattern for 'var rgGames = [...];'
    pattern = r'var rgGames = (\[.*?\]);'
    match = re.search(pattern, html_content, re.DOTALL)
    
    if match:
        try:
            games_json = match.group(1)
            games = json.loads(games_json)
            df = pd.DataFrame(games)
            # rgGames usually has keys like 'appid', 'name', 'hours_forever' (as string)
            if 'hours_forever' in df.columns:
                # Convert hours to minutes to match API format
                df['playtime_forever'] = pd.to_numeric(df['hours_forever'].str.replace(',', ''), errors='coerce').fillna(0) * 60
            return df
        except Exception as e:
            print(f"Error parsing JSON from HTML: {e}")
    else:
        print("Could not find 'rgGames' variable in HTML content.")
    
    return None

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scraping/get_user_stats.py <steamid_or_vanity_url> [path_to_reviews_html]")
        print("  python scraping/get_user_stats.py --html <path_to_library_html>")
        sys.exit(1)
        
    reviews_html_path = None
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("--"):
        reviews_html_path = sys.argv[2]

    if sys.argv[1] == "--html":
        # ... (same as before)
        if len(sys.argv) < 3:
            print("Error: Missing HTML file path.")
            sys.exit(1)
        
        file_path = sys.argv[2]
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
            
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        library_df = parse_steam_library_html(html)
        if library_df is not None:
            print(f"Successfully parsed {len(library_df)} games from HTML.")
            print(library_df[['appid', 'name', 'playtime_forever']].head(10))
        else:
            print("Failed to parse library from HTML.")
        sys.exit(0)

    input_val = sys.argv[1]
    
    # 1. Handle full URLs (profiles/ID or id/VanityName)
    if "steamcommunity.com" in input_val:
        # Match profiles/(\d+)
        profile_match = re.search(r'profiles/(\d+)', input_val)
        if profile_match:
            input_val = profile_match.group(1)
        else:
            # Match id/(\w+)
            vanity_match = re.search(r'id/([^/]+)', input_val)
            if vanity_match:
                input_val = vanity_match.group(1).rstrip('/')
    
    # 2. Try to treat as SteamID (17-digit number starting with 76)
    if input_val.isdigit() and (len(input_val) == 17 or input_val.startswith('76')):
        steamid = input_val
    else:
        print(f"Attempting to resolve vanity URL: {input_val}")
        steamid = resolve_vanity_url(input_val)
        
    if steamid:
        print(f"Fetching library for SteamID: {steamid}")
        library_df = fetch_user_library(steamid)
        if library_df is not None:
            print(f"Found {len(library_df)} games.")
            
            # Fetch reviews
            print(f"Fetching reviews for SteamID: {steamid}...")
            if reviews_html_path and os.path.exists(reviews_html_path):
                print(f"Parsing reviews from local HTML: {reviews_html_path}")
                with open(reviews_html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                # Reuse the logic but from soup
                reviews_df = fetch_user_reviews_from_soup(soup)
            else:
                reviews_df = fetch_user_reviews(steamid)
            
            if reviews_df is not None and not reviews_df.empty:
                print(f"Found {len(reviews_df)} reviews.")
                # Merge reviews into library
                library_df = library_df.merge(reviews_df, on='appid', how='left')
            else:
                library_df['user_voted_up'] = np.nan
                library_df['user_review_text'] = ""

            # Save to temp file for verification
            output_path = f"data/user_{steamid}_library.csv"
            library_df.to_csv(output_path, index=False)
            print(f"Library and reviews saved to {output_path}")
            print(library_df.head(10))
        else:
            print("Failed to fetch library.")
    else:
        print("Could not obtain a valid SteamID.")
