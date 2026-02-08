import requests
import json
import time
import os
import argparse
import logging
from datetime import datetime
import sys
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scraping.get_steam_appids as get_steam_appids
from tqdm import tqdm
from common.constants import (
    MAX_ERROR_RETRIES, ERROR_IDS_FILE, 
    SCRAPE_LOG_FILE, SCRAPE_SLEEP_TIME, SCRAPE_BACKOFF_BASE_DELAY, 
    SCRAPE_BACKOFF_MAX_RETRIES, RAW_DOWNLOAD_PATH, RAW_DOWNLOAD_REVIEWS_PATH
)
from scraping.scrape_steam import (
    get_storefront_data, get_review_stats, get_app_reviews
)

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

def download_steam_data(appids_file="data/steam_appids.csv", skipped_file="data/skipped_ids.csv", refresh=False, no_reviews=False, verbose=False):
    """
    Step 1: Focuses purely on downloading raw HTML/JSON to the local cache.
    """
    if not os.path.exists(appids_file):
        logger.info(f"{appids_file} not found. Running get_steam_appids.py first...")
        get_steam_appids.fetch_and_save_appids(appids_file)
    
    app_list_df = pd.read_csv(appids_file)
    app_ids = app_list_df['appid'].tolist()

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

    pending_ids = [aid for aid in app_ids if aid not in skipped_ids and error_counts.get(aid, 0) < MAX_ERROR_RETRIES]
    
    pbar = tqdm(pending_ids, desc="Downloading Steam Data", unit="game")
    for app_id in pbar:
        # 1. Storefront Data
        store_data = get_storefront_data(app_id, refresh=refresh, verbose=verbose)
        if store_data is False:
            skipped_ids.add(app_id)
            continue
        if store_data is None:
            error_counts[app_id] = error_counts.get(app_id, 0) + 1
            continue
            
        # 2. Review Stats
        get_review_stats(app_id, refresh=refresh, verbose=verbose)
        
        # 3. Reviews (limited pages for metadata/vibe)
        if not no_reviews:
            get_app_reviews(app_id, max_pages=10, refresh=refresh, verbose=verbose)
        
        time.sleep(SCRAPE_SLEEP_TIME)

    # Save final skipped/error counts
    pd.DataFrame({'appid': list(skipped_ids)}).to_csv(skipped_file, index=False)
    if error_counts:
        error_df = pd.DataFrame([{'appid': aid, 'error_count': count} for aid, count in error_counts.items()])
        error_df.to_csv(ERROR_IDS_FILE, index=False)

if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser(description="Step 1: Download raw Steam data to cache.")
    parser.add_argument("--ids", default="data/steam_appids.csv", help="Input AppID CSV filename.")
    parser.add_argument("--skipfile", default="data/skipped_ids.csv", help="File to store DLC/Utility IDs.")
    parser.add_argument("--refresh", action="store_true", help="Re-download all data.")
    parser.add_argument("--no-reviews", action="store_true", help="Skip downloading individual reviews.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    
    args = parser.parse_args()
    download_steam_data(appids_file=args.ids, skipped_file=args.skipfile, refresh=args.refresh, no_reviews=args.no_reviews, verbose=args.verbose)
