import json
import os
import time
import requests
import urllib.parse
from tqdm import tqdm

INPUT_FILE = "data/unique_terms.json"
OUTPUT_FILE = "data/validated_steam_links.json"
LOG_FILE = "data/validation_log.txt"

# Patterns to test
PATTERNS = [
    "https://store.steampowered.com/tags/en/{name}",
    "https://store.steampowered.com/genre/{name}",
    "https://store.steampowered.com/category/{name}"
]

# Mapping for common features that use IDs in the URL
FEATURE_MAPPING = {
    "Single-player": "2",
    "Multi-player": "1",
    "Co-op": "9",
    "Steam Achievements": "22",
    "Steam Cloud": "23",
    "Full controller support": "28",
    "Partial Controller Support": "18",
    "Steam Trading Cards": "29",
    "Steam Workshop": "30",
    "Shared/Split Screen": "24",
    "Online PvP": "36",
    "Shared/Split Screen PvP": "37",
    "Online Co-op": "38",
    "Shared/Split Screen Co-op": "39",
    "Remote Play Together": "44",
    "Stats": "15",
    "Steam Leaderboards": "25",
    "In-App Purchases": "35",
    "Captions available": "13",
    "Commentary available": "14",
    "Includes level editor": "17",
    "VR Support": "VR", # Steam has specialized search params for these
    "VR Supported": "VR",
    "VR Only": "VR",
    "Cross-Platform Multiplayer": "27",
    "Family Sharing": "62",
    "HDR available": "61",
    "Subtitle Options": "Subtitle_Options",
    "Adjustable Difficulty": "adjustable_difficulty",
    "Adjustable Text Size": "adjustable_text_size",
    "Custom Volume Controls": "custom_volume_controls",
    "Mouse Only Option": "mouse_only_option",
    "Keyboard Only Option": "keyboard_only_option",
    "Save Anytime": "save_anytime",
    "Playable without Timed Input": "playable_without_timed_input",
    "Includes Source SDK": "16",
    "Valve Anti-Cheat enabled": "8",
    "LAN PvP": "47",
    "LAN Co-op": "48",
    "Remote Play on Phone": "41",
    "Remote Play on Tablet": "42",
    "Remote Play on TV": "43",
    "Steam Turn Notifications": "51",
    "SteamVR Collectibles": "52",
    "Chat Speech-to-text": "chat_speech_to_text",
    "Chat Text-to-speech": "chat_text_to_speech"
}

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(full_msg + "\n")

def is_valid_page(url):
    """
    Checks if a Steam page is valid.
    Steam often redirects to the home page or a search page if a tag is invalid.
    We consider it valid if it returns 200 and the URL didn't redirect to the home page.
    """
    try:
        # Use a real user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # We want to follow redirects to see where we end up
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        
        if response.status_code != 200:
            return False, response.url
            
        # If it redirects to the home page (store.steampowered.com/), it's invalid
        final_url = response.url.split('?')[0].rstrip('/')
        home_page = "https://store.steampowered.com"
        
        if final_url == home_page or final_url == home_page + "/search":
            return False, response.url
            
        return True, response.url
    except Exception as e:
        log(f"Error checking {url}: {e}")
        return False, None

def validate_term(term):
    # Check if it's a known feature with an ID
    if term in FEATURE_MAPPING:
        val = FEATURE_MAPPING[term]
        if val == "VR":
            return "https://store.steampowered.com/search/?vrsupport=401"
        if val.isdigit():
            return f"https://store.steampowered.com/search/?category2={val}"
        # If it's a string, try it as a category slug
        return f"https://store.steampowered.com/category/{val}"

    # Steam often uses slug-style names: "Free to Play" -> "Free%20to%20Play" or "free_to_play"
    
    variants = [
        urllib.parse.quote(term), # Raw encoded
        urllib.parse.quote(term.lower()), # Lowercase
        term.lower().replace(' ', '_'), # lowercase_underscores (The one we missed!)
        term.replace(' ', '_'), # Spaces to underscores
        term.replace(' ', '-'), # Spaces to hyphens
        term.lower().replace(' ', ''), # Lowercase no spaces
        term.lower().replace(' ', '-').replace('-player', 'player') # "Single-player" -> "singleplayer"
    ]
    
    # Prioritize patterns: tags/en/ is the most robust catch-all
    ordered_patterns = [
        "https://store.steampowered.com/tags/en/{name}",
        "https://store.steampowered.com/genre/{name}",
        "https://store.steampowered.com/category/{name}"
    ]

    seen_variants = set()
    for variant in variants:
        if variant in seen_variants: continue
        seen_variants.add(variant)
        
        for pattern in ordered_patterns:
            url = pattern.format(name=variant)
            valid, final_url = is_valid_page(url)
            if valid:
                return url
                
    return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run extract_terms.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    all_terms = []
    for category in ['genres', 'tags', 'categories']:
        for term in data.get(category, []):
            all_terms.append((category, term))

    # Load existing progress
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            results = json.load(f)

    log(f"Starting validation of {len(all_terms)} terms...")
    
    # Filter out already processed terms, but retry those that FAILED (are None)
    to_process = [t for t in all_terms if t[1] not in results or results[t[1]] is None]
    log(f"{len(to_process)} terms remaining to process (including retries for failures).")

    pbar = tqdm(to_process)
    for cat, term in pbar:
        pbar.set_description(f"Validating {term}")
        
        valid_url = validate_term(term)
        results[term] = valid_url
        
        if valid_url:
            log(f"SUCCESS: '{term}' -> {valid_url}")
        else:
            log(f"FAILED:  '{term}'")
            
        # Save every 5 terms to prevent data loss
        if len(results) % 5 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)
                
        # Obey rate limit: 2 seconds per term (Steam TOS)
        # We made up to 6 requests per term (3 patterns * 2 variants)
        # To be safe, let's sleep 2 seconds *number of requests made*?
        # Or just a fixed 2 seconds if we only tried a few.
        # Let's do 2 seconds fixed sleep minimum, plus 1s per additional attempt.
        time.sleep(2)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    log("Validation complete.")

if __name__ == "__main__":
    main()
