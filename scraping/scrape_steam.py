import requests
import json
import time
import html
import pandas as pd
import os
import urllib.parse
import re
import argparse
import csv
import logging
import shutil
import signal
from datetime import datetime, timedelta
import sys
# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scraping.get_steam_appids as get_steam_appids
from tqdm import tqdm
from langdetect import detect, DetectorFactory
from common.constants import (
    API_KEY, CHECKPOINT_INTERVAL, MAX_ERROR_RETRIES, ERROR_IDS_FILE, 
    SCRAPE_LOG_FILE, SCRAPE_SLEEP_TIME, SCRAPE_BACKOFF_BASE_DELAY, 
    SCRAPE_BACKOFF_MAX_RETRIES, RAW_DOWNLOAD_PATH, RAW_DOWNLOAD_REVIEWS_PATH,
    ARCHIVE_PATH, SCRAPE_INPROGRESS_SUFFIX, SCRAPE_ARCHIVE_CSV_DIR
)

# Ensure reproducible language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(SCRAPE_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Session timestamp for archiving
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class SessionManager:
    """
    Handles atomic writes to the output CSV files and tracks progress.
    """
    def __init__(self, games_file, reviews_file, checkpoint_file="scraping/checkpoint_state.json"):
        self.games_file = games_file
        self.reviews_file = reviews_file
        self.checkpoint_file = checkpoint_file
        self.results = []
        self.reviews = []
        self.processed_ids = set()
        self.last_app_id = None
        
        self._load_existing_state()

    def _load_existing_state(self):
        if os.path.exists(self.games_file):
            try:
                df = pd.read_csv(self.games_file)
                self.results = df.to_dict('records')
                self.processed_ids = set(df['appid'].tolist())
                logger.info(f"Loaded {len(self.processed_ids)} processed IDs from {self.games_file}")
            except Exception as e:
                logger.error(f"Error loading {self.games_file}: {e}")

        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.last_app_id = checkpoint.get('last_app_id')
                    logger.info(f"Last checkpoint AppID: {self.last_app_id}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")

    def add_game(self, game_data):
        self.results.append(game_data)
        self.processed_ids.add(game_data['appid'])
        self.last_app_id = game_data['appid']

    def add_reviews(self, new_reviews):
        self.reviews.extend(new_reviews)

    def save_checkpoint(self):
        """
        Atomics save to CSVs and update the checkpoint file.
        Uses a retry loop to handle transient Windows file locking issues.
        """
        if not self.results and not self.reviews:
            return

        logger.info("Saving checkpoint...")
        
        max_retries = 5
        retry_delay = 1.0

        # Save games
        if self.results:
            temp_games = self.games_file + ".tmp"
            pd.DataFrame(self.results).to_csv(temp_games, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
            
            success = False
            for attempt in range(max_retries):
                try:
                    # os.replace is atomic on most systems, but Windows can be finicky if 
                    # another process (like a cloud sync or virus scanner) has a lock.
                    if os.path.exists(self.games_file):
                        # On Windows, os.replace might fail if the file is open.
                        # We try to remove first or just use os.replace and catch.
                        pass
                    os.replace(temp_games, self.games_file)
                    success = True
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Permission denied when replacing {self.games_file} (attempt {attempt+1}). Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to replace {self.games_file} after {max_retries} attempts: {e}")
            
            if not success and os.path.exists(temp_games):
                logger.info(f"Retaining temporary file {temp_games} for manual recovery.")

        # Save reviews (append)
        if self.reviews:
            file_exists = os.path.isfile(self.reviews_file)
            df = pd.DataFrame(self.reviews)
            
            success = False
            for attempt in range(max_retries):
                try:
                    df.to_csv(self.reviews_file, mode='a', index=False, header=not file_exists, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
                    success = True
                    self.reviews = []
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Permission denied when appending to {self.reviews_file} (attempt {attempt+1}). Retrying...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to append to {self.reviews_file} after {max_retries} attempts: {e}")

        # Save checkpoint state
        with open(self.checkpoint_file, 'w') as f:
            json.dump({'last_app_id': self.last_app_id, 'timestamp': datetime.now().isoformat()}, f)
        
        logger.info("Checkpoint saved.")

def backoff_request(url, max_retries=SCRAPE_BACKOFF_MAX_RETRIES, base_delay=SCRAPE_BACKOFF_BASE_DELAY, verbose=False):
    """
    Perform a GET request with exponential backoff on 429 errors.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                delay = base_delay * (2 ** retries)
                logger.warning(f"Rate limited (429) for {url}. Sleeping for {delay} seconds...")
                time.sleep(delay)
                retries += 1
            else:
                logger.error(f"Request failed for {url} with status code {response.status_code}")
                return response # Return it anyway to let the caller handle other error codes
        except Exception as e:
            logger.error(f"Exception during request to {url}: {e}")
            retries += 1
            time.sleep(base_delay)
    
    logger.error(f"Max retries exceeded for {url}")
    return None

def clean_text(text):
    """
    Remove HTML tags, decode HTML entities, strip control characters, 
    and normalize all whitespace (newlines, tabs, etc.) to spaces.
    Ensures the text is unconditionally safe for CSV format without breaking lines
    and free of encoding artifacts.
    """
    if not text:
        return ""
    
    # 1. Decode HTML entities (e.g., & -> &, " -> ")
    text = html.unescape(str(text))
    
    # 2. Remove HTML tags
    clean_re = re.compile('<.*?>')
    text = re.sub(clean_re, '', text)
    
    # 3. Normalize whitespace (newlines, tabs, etc. to a single space)
    text = re.sub(r'[\r\n\t\f\v\u2028\u2029]+', ' ', text)
    
    # 4. Remove non-printable control characters while keeping spaces
    text = "".join(char for char in text if ord(char) >= 32 or char == ' ')
    
    # 5. Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)
    
    # 6. Normalize Unicode (handle smart quotes, dashes, etc.)
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    text = text.replace('—', '-').replace('–', '-')
    
    # 7. Attempt to fix common moji-bake / double-encoding issues
    max_fixes = 3
    for _ in range(max_fixes):
        try:
            re_encoded = text.encode('cp1252').decode('utf-8')
            if re_encoded == text: break
            text = re_encoded
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                re_encoded = text.encode('latin-1').decode('utf-8')
                if re_encoded == text: break
                text = re_encoded
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
    
    return text.strip()

def is_english(text):
    """
    Check if the provided text is primarily in English.
    """
    if not text or len(text.strip()) < 10:
        return False
    try:
        return detect(text) == 'en'
    except:
        return False

def format_seconds(seconds):
    """
    Convert seconds into a human-readable string (Hh Mm Ss).
    """
    return str(timedelta(seconds=int(seconds)))

def move_to_archive(app_id, suffix, ext, base_path):
    """
    Move an existing cached file to the session's archive directory.
    """
    if not ARCHIVE_PATH:
        return
    
    source_file = os.path.join(base_path, f"{app_id}_{suffix}.{ext}")
    if os.path.exists(source_file):
        try:
            # Create session archive folder
            archive_session_dir = os.path.join(ARCHIVE_PATH, SESSION_TIMESTAMP)
            # Maintain reviews subfolder in archive if needed
            if base_path == RAW_DOWNLOAD_REVIEWS_PATH:
                archive_session_dir = os.path.join(archive_session_dir, "reviews")
            
            os.makedirs(archive_session_dir, exist_ok=True)
            
            dest_file = os.path.join(archive_session_dir, f"{app_id}_{suffix}.{ext}")
            shutil.move(source_file, dest_file)
            # logger.info(f"Archived {app_id}_{suffix}.{ext} to {SESSION_TIMESTAMP}")
        except Exception as e:
            logger.error(f"Error archiving {app_id} ({suffix}): {e}")

def get_cached_file(app_id, suffix, ext, base_path=RAW_DOWNLOAD_PATH):
    """
    Check if a cached version of the requested data exists.
    """
    if not base_path:
        return None
    
    file_path = os.path.join(base_path, f"{app_id}_{suffix}.{ext}")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading cache for {app_id} ({suffix}): {e}")
    return None

def save_to_cache(app_id, suffix, ext, content, base_path=RAW_DOWNLOAD_PATH):
    """
    Save raw response content to the local cache directory.
    """
    if not base_path:
        return
    
    try:
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        
        file_path = os.path.join(base_path, f"{app_id}_{suffix}.{ext}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Error saving cache for {app_id} ({suffix}): {e}")

def get_storefront_data(app_id, refresh=False, verbose=False):
    """
    Fetch comprehensive game data directly from the Steam Storefront HTML.
    """
    url = f"https://store.steampowered.com/app/{app_id}/?l=english"
    cookies = {
        'birthtime': '283996801', 
        'wants_mature_content': '1'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    if verbose:
        logger.info(f"[Storefront Request]: {url}")

    # Archiving logic
    if refresh:
        move_to_archive(app_id, "storefront", "html", RAW_DOWNLOAD_PATH)

    # Check cache first
    cached_html = None if refresh else get_cached_file(app_id, "storefront", "html")
    
    if cached_html:
        if verbose: logger.info(f"Using cached storefront for {app_id}")
        html_content = cached_html
    else:
        # Perform download with backoff
        retries = 0
        response = None
        while retries < SCRAPE_BACKOFF_MAX_RETRIES:
            try:
                response = requests.get(url, headers=headers, cookies=cookies, timeout=15)
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    delay = SCRAPE_BACKOFF_BASE_DELAY * (2 ** retries)
                    logger.warning(f"Rate limited (429) for {url}. Sleeping for {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
                elif response.status_code == 404:
                    return False
                else:
                    logger.error(f"Request failed for {url} with status code {response.status_code}")
                    return None
            except Exception as e:
                logger.error(f"Exception during request to {url}: {e}")
                retries += 1
                time.sleep(SCRAPE_BACKOFF_BASE_DELAY)

        if not response or response.status_code != 200:
            return None

        html_content = response.text
        # Save to cache after successful download
        save_to_cache(app_id, "storefront", "html", html_content)
    
    # Check if we were redirected to the home page (app doesn't exist)
    if 'id="home_main_content"' in html_content or ('id="appHubAppName"' not in html_content and 'class="apphub_AppName"' not in html_content):
        return False

    data = {'appid': app_id}

    # 1. Name
    name_match = re.search(r'<div[^>]*class="apphub_AppName"[^>]*>([^<]*)</div>', html_content)
    data['name'] = clean_text(name_match.group(1)) if name_match else "Unknown"

    # 2. Release Date
    date_match = re.search(r'<div class="date">([^<]*)</div>', html_content)
    data['release_date'] = clean_text(date_match.group(1)) if date_match else ""

    # 3. Description (Short)
    desc_match = re.search(r'<div class="game_description_snippet">([^<]*)</div>', html_content, re.DOTALL)
    data['short_description'] = clean_text(desc_match.group(1)) if desc_match else ""

    # 4. Detailed Description
    detail_match = re.search(r'<div id="game_area_description" class="game_area_description">.*?</div>', html_content, re.DOTALL)
    if not detail_match:
        detail_match = re.search(r'<div class="game_area_description">.*?</div>', html_content, re.DOTALL)
    
    data['detailed_description'] = clean_text(detail_match.group(0)) if detail_match else ""
    data['about_the_game'] = data['detailed_description']

    # 5. Tags (High Fidelity JSON)
    tags_match = re.search(r'InitAppTagModal\(\s*\d+,\s*(\[.*?\])\s*,', html_content, re.DOTALL)
    tags_dict = {}
    if tags_match:
        try:
            tags_list = json.loads(tags_match.group(1))
            tags_dict = {t['name']: t['count'] for t in tags_list}
        except:
            pass
    data['tags'] = str(tags_dict)

    # 6. Price
    price_match = re.search(r'<div class="game_purchase_price price" [^>]*>([^<]*)</div>', html_content)
    if not price_match:
        price_match = re.search(r'<div class="discount_final_price">([^<]*)</div>', html_content)
    data['price'] = clean_text(price_match.group(1)) if price_match else ""

    # 7. Developer / Publisher
    dev_match = re.search(r'<b>Developer:</b>.*?href="[^"]*">([^<]*)</a>', html_content, re.DOTALL)
    pub_match = re.search(r'<b>Publisher:</b>.*?href="[^"]*">([^<]*)</a>', html_content, re.DOTALL)
    data['developers'] = clean_text(dev_match.group(1)) if dev_match else ""
    data['publishers'] = clean_text(pub_match.group(1)) if pub_match else ""

    # 8. Genres
    genre_section = re.search(r'<b>Genre:</b>(.*?)</div>', html_content, re.DOTALL)
    if genre_section:
        genres = re.findall(r'<a[^>]*>([^<]*)</a>', genre_section.group(1))
        data['genres'] = ",".join([clean_text(g) for g in genres])
    else:
        data['genres'] = ""

    # 9. Platforms
    data['windows'] = 'platform_img win' in html_content
    data['mac'] = 'platform_img mac' in html_content
    data['linux'] = 'platform_img linux' in html_content

    # 10. Metadata / Categories
    categories = re.findall(r'<a class="name" href="https://store.steampowered.com/search/\?(?:category2|category3)=\d+">([^<]*)</a>', html_content)
    data['categories'] = ",".join(set(categories))

    # 11. Languages
    lang_table = re.search(r'<table class="game_language_options".*?>(.*?)</table>', html_content, re.DOTALL)
    if lang_table:
        langs = re.findall(r'<td class="ellipsis">\s*([^<]*)\s*</td>', lang_table.group(1))
        data['supported_languages'] = ",".join([clean_text(l) for l in langs])
    else:
        data['supported_languages'] = ""

    # 12. Mature Content (Adult Only Detection)
    data['mature_content'] = 1 if "Adult Only" in html_content else 0
    data['mature_notes'] = "" 

    # 13. Breadcrumbs (Software/Game Detection)
    breadcrumb_section = re.search(r'<div class="blockbg">.*?</div>', html_content, re.DOTALL)
    if breadcrumb_section:
        bc_links = re.findall(r'<a[^>]*>([^<]*)</a>', breadcrumb_section.group(0))
        breadcrumbs = [l.strip() for l in bc_links]
        if breadcrumbs and breadcrumbs[0] == "All Software":
            if data['genres']:
                data['genres'] += ",Software"
            else:
                data['genres'] = "Software"

    # 14. Image / Media
    header_match = re.search(r'<img class="game_header_image_full" [^>]*src="([^"]*)"', html_content)
    if not header_match:
        header_match = re.search(r'<img class="package_header" [^>]*src="([^"]*)"', html_content)
    data['header_image'] = header_match.group(1) if header_match else ""
    
    ss_matches = re.findall(r'https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/\d+/ss_[^?"]*', html_content)
    data['screenshots'] = ",".join(set(ss_matches))

    # 15. Metacritic
    meta_match = re.search(r'<div id="game_area_metascore"[^>]*>.*?<span class="score">(\d+)</span>', html_content, re.DOTALL)
    data['metacritic_score'] = meta_match.group(1) if meta_match else 0
    meta_url = re.search(r'<div id="game_area_metascore"[^>]*>.*?<a href="([^"]*)"', html_content, re.DOTALL)
    data['metacritic_url'] = meta_url.group(1) if meta_url else ""

    # 16. Achievements
    ach_match = re.search(r'View all (\d+) achievements', html_content)
    data['achievements'] = ach_match.group(1) if ach_match else 0

    return data

def get_review_stats(app_id, refresh=False, verbose=False):
    """
    Fetch the review summary, prioritizing English reviews.
    Falls back to all languages if no English reviews are found.
    """
    # 1. Try English
    if refresh:
        move_to_archive(app_id, "stats_english", "json", RAW_DOWNLOAD_PATH)
    
    # Check cache for English
    cached_json_en = None if refresh else get_cached_file(app_id, "stats_english", "json")
    if cached_json_en:
        try:
            data = json.loads(cached_json_en)
            if data.get('total_reviews', 0) > 0:
                return data
        except:
            pass

    # Fetch English
    url_en = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=english"
    response_en = backoff_request(url_en, verbose=verbose)
    if response_en and response_en.status_code == 200:
        try:
            data = response_en.json()
            if data.get('success'):
                summary = data.get('query_summary', {})
                if summary.get('total_reviews', 0) > 0:
                    save_to_cache(app_id, "stats_english", "json", json.dumps(summary))
                    return summary
        except Exception as e:
            logger.error(f"Error parsing English Review Stats JSON for {app_id}: {e}")

    # 2. Fallback to All (Legacy/Default)
    if refresh:
        move_to_archive(app_id, "stats", "json", RAW_DOWNLOAD_PATH)

    cached_json_all = None if refresh else get_cached_file(app_id, "stats", "json")
    if cached_json_all:
        try:
            return json.loads(cached_json_all)
        except:
            pass

    url_all = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=all"
    response_all = backoff_request(url_all, verbose=verbose)
    if response_all and response_all.status_code == 200:
        try:
            data = response_all.json()
            if data.get('success'):
                summary = data.get('query_summary', {})
                if summary:
                    save_to_cache(app_id, "stats", "json", json.dumps(summary))
                return summary
        except Exception as e:
            logger.error(f"Error parsing All Review Stats JSON for {app_id}: {e}")
    
    return {}

def get_app_reviews(app_id, max_pages=10, existing_review_ids=None, refresh=False, verbose=False):
    """
    Fetch user reviews from the Steam API with pagination.
    """
    all_reviews = []
    cursor = '*'
    one_year_ago = (datetime.now() - timedelta(days=365)).timestamp()
    date_collected = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing_review_ids = existing_review_ids or set()
    
    for page in range(max_pages):
        cache_suffix = f"reviews_p{page}"
        
        if refresh:
            move_to_archive(app_id, cache_suffix, "json", RAW_DOWNLOAD_REVIEWS_PATH)

        cached_json = None if refresh else get_cached_file(app_id, cache_suffix, "json", base_path=RAW_DOWNLOAD_REVIEWS_PATH)
        
        data = None
        try:
            if cached_json:
                data = json.loads(cached_json)
            else:
                encoded_cursor = urllib.parse.quote(cursor)
                url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&num_per_page=100&cursor={encoded_cursor}&filter=recent&language=english&day_range=365"
                
                response = backoff_request(url, verbose=verbose)
                if response and response.status_code == 200:
                    data = response.json()
                    if data and data.get('success'):
                        save_to_cache(app_id, cache_suffix, "json", json.dumps(data), base_path=RAW_DOWNLOAD_REVIEWS_PATH)
                
            if data and data.get('success'):
                reviews = data.get('reviews', [])
                if not reviews:
                    break
                for r in reviews:
                    review_id = r.get('recommendationid')
                    if review_id in existing_review_ids:
                        if verbose:
                            logger.info(f"Encountered existing review {review_id}. Terminating review fetch for {app_id}.")
                        return all_reviews
                        
                    timestamp_created = r.get('timestamp_created', 0)
                    if timestamp_created >= one_year_ago:
                        rev_text = r.get('review', '')
                        if is_english(rev_text):
                            author = r.get('author', {})
                            review_data = {
                                'appid': app_id,
                                'review_id': r.get('recommendationid'),
                                'author_id': author.get('steamid'),
                                'review_text': clean_text(rev_text),
                                'timestamp_created': timestamp_created,
                                'timestamp_updated': r.get('timestamp_updated'),
                                'voted_up': r.get('voted_up'),
                                'votes_up': r.get('votes_up'),
                                'votes_funny': r.get('votes_funny'),
                                'weighted_vote_score': r.get('weighted_vote_score'),
                                'comment_count': r.get('comment_count'),
                                'steam_purchase': r.get('steam_purchase'),
                                'received_for_free': r.get('received_for_free'),
                                'written_during_early_access': r.get('written_during_early_access'),
                                'author_num_games_owned': author.get('num_games_owned'),
                                'author_num_reviews': author.get('num_reviews'),
                                'author_playtime_forever': author.get('playtime_forever'),
                                'author_playtime_last_two_weeks': author.get('playtime_last_two_weeks'),
                                'author_playtime_at_review': author.get('playtime_at_review'),
                                'author_last_played': author.get('last_played'),
                                'date_collected': date_collected
                            }
                            all_reviews.append(review_data)
                    else:
                        return all_reviews
                
                new_cursor = data.get('cursor')
                if not new_cursor or new_cursor == cursor:
                    break
                cursor = new_cursor
                # Delay if we just performed a network call
                if not cached_json:
                    time.sleep(1.0)
            else:
                break
        except Exception as e:
            if verbose:
                logger.error(f"Error fetching reviews for {app_id}: {e}")
            break
            
    return all_reviews

def save_reviews_to_csv(reviews, reviews_file):
    """
    Append reviews to a CSV file using utf-8-sig encoding.
    """
    if not reviews:
        return
    
    file_exists = os.path.isfile(reviews_file)
    df = pd.DataFrame(reviews)
    df.to_csv(reviews_file, mode='a', index=False, header=not file_exists, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')

def scrape_games(output_file="scraped_games.csv", reviews_file="scraped_reviews.csv", appids_file="data/steam_appids.csv", skipped_file="data/skipped_ids.csv", checkpoint_interval=20, refresh=False, no_reviews=False, verbose=False):
    """
    Main scraping orchestration logic.
    """
    if not os.path.exists(appids_file):
        logger.info(f"{appids_file} not found. Running get_steam_appids.py first...")
        get_steam_appids.fetch_and_save_appids(appids_file)
    
    app_list_df = pd.read_csv(appids_file)
    app_ids = app_list_df['appid'].tolist()
    
    # -------------------------------------------------------------------------
    # Use temporary "in-progress" files to avoid locking/overwriting production
    # data while the scrape is running.
    # -------------------------------------------------------------------------
    inprogress_games = output_file.replace('.csv', SCRAPE_INPROGRESS_SUFFIX)
    inprogress_reviews = reviews_file.replace('.csv', SCRAPE_INPROGRESS_SUFFIX)

    # If in-progress files don't exist, initialize them from the last successful scrape
    # so we can resume where we left off (or just start fresh if no previous scrape).
    if not os.path.exists(inprogress_games) and os.path.exists(output_file):
        logger.info(f"Initializing {inprogress_games} from {output_file}...")
        try:
            shutil.copy2(output_file, inprogress_games)
        except Exception as e:
            logger.error(f"Failed to copy {output_file} to {inprogress_games}: {e}")

    if not os.path.exists(inprogress_reviews) and os.path.exists(reviews_file):
        logger.info(f"Initializing {inprogress_reviews} from {reviews_file}...")
        try:
            shutil.copy2(reviews_file, inprogress_reviews)
        except Exception as e:
            logger.error(f"Failed to copy {reviews_file} to {inprogress_reviews}: {e}")

    session = SessionManager(inprogress_games, inprogress_reviews)

    # Load existing review IDs for deduplication (from the in-progress file now)
    existing_review_ids = set()
    # We check the in-progress file first, as it's the active one
    target_reviews_load = inprogress_reviews if os.path.exists(inprogress_reviews) else reviews_file
    
    if os.path.exists(target_reviews_load):
        try:
            rev_ids_df = pd.read_csv(target_reviews_load, usecols=['review_id'])
            existing_review_ids = set(rev_ids_df['review_id'].tolist())
            logger.info(f"Loaded {len(existing_review_ids)} existing review IDs from {target_reviews_load}.")
        except Exception as e:
            logger.warning(f"Could not load review IDs from {target_reviews_load}: {e}")

    # Load skipped IDs
    skipped_ids = set()
    if os.path.exists(skipped_file):
        try:
            skipped_df = pd.read_csv(skipped_file)
            skipped_ids = set(skipped_df['appid'].tolist())
            logger.info(f"Loaded {len(skipped_ids)} previously skipped IDs.")
        except:
            pass

    # Load persistent error IDs
    error_counts = {}
    if os.path.exists(ERROR_IDS_FILE):
        try:
            error_df = pd.read_csv(ERROR_IDS_FILE)
            error_counts = dict(zip(error_df['appid'], error_df['error_count']))
            logger.info(f"Loaded {len(error_counts)} persistent error IDs.")
        except:
            pass

    # Signal handling for graceful exit
    def signal_handler(sig, frame):
        logger.info("Signal received. Saving progress before exiting...")
        session.save_checkpoint()
        # Save skipped/error files too
        pd.DataFrame({'appid': list(skipped_ids)}).to_csv(skipped_file, index=False)
        if error_counts:
            error_df = pd.DataFrame([{'appid': aid, 'error_count': count} for aid, count in error_counts.items()])
            error_df.to_csv(ERROR_IDS_FILE, index=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    actually_processed_since_start = 0
    processed_count_since_checkpoint = 0

    # If refreshing, we process all IDs that aren't skipped or errored out
    # If not refreshing, we only process IDs that aren't already in results
    pending_ids = [aid for aid in app_ids if (refresh or aid not in session.processed_ids) and aid not in skipped_ids and error_counts.get(aid, 0) < MAX_ERROR_RETRIES]
    
    while pending_ids:
        logger.info(f"Starting pass with {len(pending_ids)} pending games")
        errored_ids = []
        
        pbar = tqdm(pending_ids, desc="Scraping Steam", unit="game")
        for app_id in pbar:
            if verbose:
                logger.info(f"Processing {app_id}...")
            
            # 1. Fetch Storefront Data
            store_data = get_storefront_data(app_id, refresh=refresh, verbose=verbose)
            time.sleep(0.5)

            if store_data is False:
                skipped_ids.add(app_id)
                actually_processed_since_start += 1
                processed_count_since_checkpoint += 1
                continue

            if store_data is None:
                error_counts[app_id] = error_counts.get(app_id, 0) + 1
                actually_processed_since_start += 1
                processed_count_since_checkpoint += 1
                if error_counts[app_id] < MAX_ERROR_RETRIES:
                    errored_ids.append(app_id)
                time.sleep(SCRAPE_SLEEP_TIME)
                continue

            # 2. Filter DLC/Utility
            genres_list = store_data.get('genres', '').split(',')
            is_utility = 'Software' in genres_list or 'Utilities' in genres_list
            is_dlc = store_data.get('categories', '').find('Downloadable Content') != -1 or store_data.get('price', '').lower().find('dlc') != -1
            
            if is_dlc or is_utility:
                skipped_ids.add(app_id)
                actually_processed_since_start += 1
                processed_count_since_checkpoint += 1
                continue

            # 3. Process reviews
            if not no_reviews:
                reviews = get_app_reviews(app_id, existing_review_ids=existing_review_ids, refresh=refresh, verbose=verbose)
                session.add_reviews(reviews)
                for r in reviews:
                    existing_review_ids.add(r['review_id'])
                
                time.sleep(0.5)
            review_stats = get_review_stats(app_id, refresh=refresh, verbose=verbose)
            
            game_data = store_data.copy()
            game_data['positive'] = review_stats.get('total_positive', 0)
            game_data['negative'] = review_stats.get('total_negative', 0)
            game_data['user_score'] = review_stats.get('review_score', 0)
            game_data['owners'] = "0 .. 20,000"
            game_data['average_forever'] = 0
            game_data['median_forever'] = 0
            game_data['is_dlc'] = is_dlc
            game_data['recommendations'] = review_stats.get('total_reviews', 0)
            game_data['movies'] = "" 
            game_data['required_age'] = 0
            
            session.add_game(game_data)
            actually_processed_since_start += 1
            processed_count_since_checkpoint += 1
            
            time.sleep(SCRAPE_SLEEP_TIME)
            
            if processed_count_since_checkpoint >= checkpoint_interval:
                session.save_checkpoint()
                pd.DataFrame({'appid': list(skipped_ids)}).to_csv(skipped_file, index=False)
                processed_count_since_checkpoint = 0

        pending_ids = errored_ids
        if errored_ids:
            logger.info(f"Retrying {len(errored_ids)} errored IDs after a short sleep...")
            time.sleep(10)

    session.save_checkpoint()
    pd.DataFrame({'appid': list(skipped_ids)}).to_csv(skipped_file, index=False)
    
    if error_counts:
        error_df = pd.DataFrame([{'appid': aid, 'error_count': count} for aid, count in error_counts.items()])
        error_df.to_csv(ERROR_IDS_FILE, index=False)
        
    logger.info(f"Finished! Total apps saved: {len(session.results)}")

    # -------------------------------------------------------------------------
    # Atomic Swap: Archive old production files and promote in-progress files
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = SCRAPE_ARCHIVE_CSV_DIR
    os.makedirs(archive_dir, exist_ok=True)
    
    # 1. Archive existing production files
    if os.path.exists(output_file):
        archive_name = f"{os.path.basename(output_file).replace('.csv', '')}_{timestamp}.csv"
        archive_path = os.path.join(archive_dir, archive_name)
        logger.info(f"Archiving {output_file} to {archive_path}")
        try:
            shutil.move(output_file, archive_path)
        except Exception as e:
            logger.error(f"Failed to archive {output_file}: {e}")

    if os.path.exists(reviews_file):
        archive_name = f"{os.path.basename(reviews_file).replace('.csv', '')}_{timestamp}.csv"
        archive_path = os.path.join(archive_dir, archive_name)
        logger.info(f"Archiving {reviews_file} to {archive_path}")
        try:
            shutil.move(reviews_file, archive_path)
        except Exception as e:
            logger.error(f"Failed to archive {reviews_file}: {e}")

    # 2. Promote in-progress files to production
    if os.path.exists(inprogress_games):
        logger.info(f"Promoting {inprogress_games} to {output_file}")
        try:
            os.replace(inprogress_games, output_file)
        except Exception as e:
            logger.error(f"Failed to promote {inprogress_games}: {e}")

    if os.path.exists(inprogress_reviews):
        logger.info(f"Promoting {inprogress_reviews} to {reviews_file}")
        try:
            os.replace(inprogress_reviews, reviews_file)
        except Exception as e:
            logger.error(f"Failed to promote {inprogress_reviews}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Steam game data and reviews.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--output", default="scraped_games.csv", help="Output CSV filename for games.")
    parser.add_argument("--reviews_output", default="scraped_reviews.csv", help="Output CSV filename for reviews.")
    parser.add_argument("--ids", default="data/steam_appids.csv", help="Input AppID CSV filename.")
    parser.add_argument("--skipfile", default="data/skipped_ids.csv", help="File to store DLC/Utility IDs.")
    parser.add_argument("--checkpoint", type=int, default=20, help="Save interval.")
    parser.add_argument("--refresh", action="store_true", help="Archive existing cache and re-download all data.")
    parser.add_argument("--no-reviews", action="store_true", help="Skip downloading individual reviews.")
    
    args = parser.parse_args()
    scrape_games(output_file=args.output, reviews_file=args.reviews_output, appids_file=args.ids, skipped_file=args.skipfile, checkpoint_interval=args.checkpoint, refresh=args.refresh, no_reviews=args.no_reviews, verbose=args.verbose)
