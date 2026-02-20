#!/usr/bin/env python3
"""
Generate a random game recommendation for Discord sharing.
Mimics the "Surprise Me" button but runs from command line.
"""

import os
import sys
import json
import random
import time
import requests
import pandas as pd
import subprocess
from pathlib import Path

# Add parent directory to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.constants import BACKEND_URL

def start_backend():
    """Start the backend server as a background process."""
    print("Starting backend server...")
    root_dir = Path(__file__).parent.parent
    try:
        # Using the same command format as in deployment/run scripts
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=root_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print(f"Backend process started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"ERROR: Failed to start backend: {e}")
        return None

def wait_for_backend(max_retries=30, retry_delay=2):
    """Wait for backend server to become available, starting it if necessary."""
    print(f"Checking for backend at {BACKEND_URL}...")
    
    backend_started = False
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{BACKEND_URL}/games", timeout=2)
            if response.status_code == 200:
                print(f"Backend ready after {i+1} attempts")
                return True
        except requests.exceptions.ConnectionError:
            if not backend_started:
                print("Backend not found. Attempting to start it...")
                start_backend()
                backend_started = True
        except Exception as e:
            print(f"Error checking backend: {e}")
            
        time.sleep(retry_delay)
    
    print(f"ERROR: Backend failed to become ready after {max_retries} attempts")
    return False

def load_adjectives():
    """Load adjectives from common/common_adjectives.txt"""
    adj_path = Path(__file__).parent.parent / "common" / "common_adjectives.txt"
    try:
        with open(adj_path, "r", encoding="utf-8") as f:
            adjectives = [line.strip() for line in f if line.strip()]
        if adjectives:
            return adjectives
    except Exception as e:
        print(f"Warning: Could not load adjectives: {e}")
    
    # Fallback adjectives
    return ["exciting", "atmospheric", "challenging", "relaxing", "mysterious", "colorful",
            "epic", "cozy", "intense", "thoughtful", "hilarious", "creepy"]

def get_game_list():
    """Fetch list of all games from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/games", timeout=5)
        response.raise_for_status()
        games = response.json()
        if games:
            return games
        else:
            raise ValueError("No games returned from backend")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to backend at {BACKEND_URL}")
        print("Please ensure the backend server is running (python app/server.py or uvicorn app.server:app)")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to fetch games: {e}")
        sys.exit(1)

def generate_random_params(games):
    """Generate random recommendation parameters"""
    adjectives = load_adjectives()
    
    params = {
        "alpha": random.uniform(0, 1.0),
        "beta": random.uniform(0, 1.0),
        "quality_pref": 1.0,  # Max quality bias as specified
        "age_pref": random.uniform(-1.0, 1.0),
        "pop_pref": random.uniform(-1.0, 1.0),
        "disc_pref": 1.0,  # Max discovery (Wild Cards / Niche)
        "length_pref": random.uniform(-1.0, 1.0),
        "difficulty_pref": random.uniform(-1.0, 1.0),
        "price_pref": random.uniform(-1.0, 1.0),
        "remove_vr": True,
        "english_only": True,
        "remove_nsfw": True,
        "remove_utilities": True,
        "remove_unreleased": True,
        "top_k": 10,  # Return top 10 results
        "prompt": random.choice(adjectives),
        "seed_games": [random.choice(games)],
        "genres": []
    }
    return params

def get_recommendation(params):
    """Call backend /recommend endpoint"""
    try:
        response = requests.post(f"{BACKEND_URL}/recommend", json=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        return results
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to backend at {BACKEND_URL}")
        print("Please ensure the backend server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Recommendation request failed: {e}")
        if hasattr(response, 'status_code'):
            print(f"Status code: {response.status_code}")
            try:
                print(f"Response: {response.json()}")
            except:
                pass
        sys.exit(1)

def format_result(game, rank=1):
    """Format the recommendation for display"""
    steam_link = f"https://store.steampowered.com/app/{game['appid']}"
    
    # Format release date
    raw_date = game.get('release_date', 'Unknown')
    if raw_date and str(raw_date).strip() not in ["", "NaT", "Unknown", "<NA>"]:
        try:
            # Try to parse and reformat to "Oct 14, 2015"
            dt = pd.to_datetime(raw_date)
            release_date = dt.strftime("%b %d, %Y")
        except:
            release_date = str(raw_date)
    else:
        release_date = "Unknown"
    
    # Format estimated length (convert from minutes to hours)
    est_length = game.get('estimated_playtime', 0)
    if est_length and est_length > 0:
        hours = est_length / 60.0
        length_str = f"{hours:.1f} hours"
    else:
        length_str = "Unknown"
    
    # Format difficulty
    difficulty = game.get('difficulty_predicted')
    if difficulty is not None and not (isinstance(difficulty, float) and (pd.isna(difficulty) or difficulty == 0)):
        difficulty_str = f"{difficulty:.1f}/10"
    else:
        difficulty_str = "Unknown"
    
    # Build output
    output = []
    output.append(f"Game: {game['name']}")
    output.append(f"Steam: {steam_link}")
    output.append("")  # Empty line after link
    output.append(f"📅 Release Date: {release_date}")
    output.append(f"⏱️  Est. Length: {length_str}")
    output.append(f"🎯 Difficulty: {difficulty_str}")
    
    # Add review count and percentage
    positive = game.get('positive', 0)
    negative = game.get('negative', 0)
    total = positive + negative
    if total > 0:
        percentage = (positive / total) * 100
        output.append(f"👍 {positive:,}/{total:,} ({percentage:.1f}% positive)")
    
    # Add genres if available
    genres = game.get('genres', [])
    if genres:
        if isinstance(genres, str):
            try:
                # Handle both JSON-like lists and comma-separated strings
                if genres.startswith('['):
                    genres = json.loads(genres.replace("'", '"'))
                else:
                    genres = [g.strip() for g in genres.split(',')]
            except:
                genres = [g.strip() for g in genres.split(',')]
        output.append(f"🏷️  Genres: {', '.join(genres)}")
    
    return "\n".join(output)

def main():
    print("🎲 Generating random recommendations...")
    
    # Wait for backend to be ready
    if not wait_for_backend():
        print("ERROR: Backend not available")
        sys.exit(1)
    
    # Load game list
    games = get_game_list()
    
    # Generate random parameters
    params = generate_random_params(games)
    print(f"Querying with prompt: '{params['prompt']}' and seed: '{params['seed_games'][0]}'")
    print(f"Discovery: {params['disc_pref']}, Quality: {params['quality_pref']}")
    
    # Get recommendation
    results = get_recommendation(params)
    
    if not results:
        print("ERROR: No recommendations returned")
        sys.exit(1)
    
    # Display results
    print("\n" + "=" * 60)
    print("🎮 TOP 10 RANDOM RECOMMENDATIONS")
    print("=" * 60)
    
    for i, game in enumerate(results):
        print(format_result(game, rank=i+1))
        print()  # Extra newline between items
    
    print("\n" + "=" * 60)
    print("Generated by SteamJackalope Random Recommender")
    print("=" * 60)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()