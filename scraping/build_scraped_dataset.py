import json
import os
import argparse
import logging
import pandas as pd
import csv
import re
import sys
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from common.constants import (
    RAW_DOWNLOAD_PATH, RAW_DOWNLOAD_REVIEWS_PATH, SCRAPE_LOG_FILE
)
from scraping.scrape_steam import (
    clean_text, is_english
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

def parse_storefront_html(app_id, html_content):
    """
    Extracted parsing logic from scrape_steam.py.
    """
    if not html_content or 'id="appHubAppName"' not in html_content and 'class="apphub_AppName"' not in html_content:
        return None

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

    # 5. Tags
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
    # Robust Multi-Stage Price Extraction
    def to_float(s):
        if not s: return 0.0
        # Remove non-numeric chars except decimal points/commas
        s_clean = re.sub(r'[^\d,.]', '', str(s)).replace(',', '.')
        try: return float(s_clean)
        except: return 0.0

    # A. Get reliable meta price (Current Price for specific AppID)
    meta_price_val = 0.0
    meta_price_str = ""
    meta_match = re.search(r'<meta itemprop="price" content="([^"]*)">', html_content)
    if meta_match:
        meta_price_str = meta_match.group(1).strip()
        meta_price_val = to_float(meta_price_str)

    # B. Try to find the "Original" price if there's a sale
    # We look for a discount block where the final price matches our meta price
    # This ensures we get the original price for the BASE game, not a bundle.
    discount_blocks = re.findall(r'<div class="discount_original_price">([^<]*)</div>\s*<div class="discount_final_price">([^<]*)</div>', html_content)
    
    found_orig = False
    for orig_str, final_str in discount_blocks:
        if abs(to_float(final_str) - meta_price_val) < 0.01:
            data['price'] = clean_text(orig_str)
            found_orig = True
            break

    if not found_orig:
        if meta_price_val > 0:
            data['price'] = f"${meta_price_str}" if "$" not in meta_price_str else meta_price_str
        else:
            # 1. Tags Check for "Free to Play" - strongest positive indicator
            tags_regex = r'<a[^>]*class="app_tag"[^>]*>\s*([^<]*)\s*</a>'
            tags = [t.strip() for t in re.findall(tags_regex, html_content)]
            is_free_tag = "Free to Play" in tags
            
            # 2. Check for "Free" in standard purchase boxes
            price_regex = r'<div class="game_purchase_price price"[^>]*>([^<]*)</div>'
            price_match = re.search(price_regex, html_content)
            
            if is_free_tag:
                data['price'] = "Free To Play"
            elif price_match:
                val = price_match.group(1).strip()
                if val and "free" in val.lower():
                    data['price'] = "Free To Play"
                elif val:
                    data['price'] = val
                else:
                    data['price'] = "N/A"
            elif "no longer available" in html_content or "no longer available on Steam" in html_content:
                data['price'] = "Delisted"
            elif "Coming soon" in html_content or "Coming Soon" in html_content:
                data['price'] = "Coming Soon"
            elif meta_match and meta_price_val == 0:
                data['price'] = "Free"
            else:
                data['price'] = "N/A"

    # 7. Dev/Pub
    # Capture all developer/publisher links if multiple exist
    dev_section = re.search(r'<b>Developer:</b>(.*?)(?=<b>|</div>|<br>|$)', html_content, re.DOTALL)
    if dev_section:
        devs = re.findall(r'<a[^>]*>([^<]*)</a>', dev_section.group(1))
        data['developers'] = ",".join([clean_text(d) for d in devs])
    else:
        data['developers'] = ""

    pub_section = re.search(r'<b>Publisher:</b>(.*?)(?=<b>|</div>|<br>|$)', html_content, re.DOTALL)
    if pub_section:
        pubs = re.findall(r'<a[^>]*>([^<]*)</a>', pub_section.group(1))
        data['publishers'] = ",".join([clean_text(p) for p in pubs])
    else:
        data['publishers'] = ""

    # 8. Genres
    # Stop at the next <b> tag to avoid capturing Developer/Publisher links
    genre_section = re.search(r'<b>Genre:</b>(.*?)(?=<b>|</div>|<br>|$)', html_content, re.DOTALL)
    if genre_section:
        genres = re.findall(r'<a[^>]*>([^<]*)</a>', genre_section.group(1))
        data['genres'] = ",".join([clean_text(g) for g in genres])
    else:
        data['genres'] = ""

    # 9. Platforms
    data['windows'] = 'platform_img win' in html_content
    data['mac'] = 'platform_img mac' in html_content
    data['linux'] = 'platform_img linux' in html_content

    # 10. Categories
    categories = re.findall(r'<a class="name" href="https://store.steampowered.com/search/\?(?:category2|category3)=\d+">([^<]*)</a>', html_content)
    data['categories'] = ",".join(set(categories))

    # 11. Languages
    # Robust regex for table with potentially multiple spaces or attributes
    lang_table = re.search(r'<table\s+class="game_language_options".*?>(.*?)</table>', html_content, re.DOTALL)
    if lang_table:
        # Only capture rows that ARE NOT marked as "unsupported"
        # We look for rows, then check if they contain the unsupported class
        rows = re.findall(r'<tr\s+style=""\s+class="([^"]*)">.*?<td\s+style="[^"]*"\s+class="ellipsis">\s*([^<]*?)\s*</td>', lang_table.group(1), re.DOTALL)
        supported = [l.strip() for cls, l in rows if "unsupported" not in cls]
        data['supported_languages'] = ",".join([clean_text(l) for l in supported])
    else:
        data['supported_languages'] = ""

    # 12. Breadcrumbs (Software/Game Detection)
    breadcrumb_section = re.search(r'<div class="blockbg">.*?</div>', html_content, re.DOTALL)
    if breadcrumb_section:
        bc_links = re.findall(r'<a[^>]*>([^<]*)</a>', breadcrumb_section.group(0))
        breadcrumbs = [l.strip() for l in bc_links]
        if breadcrumbs and breadcrumbs[0] == "All Software":
            if data['genres']:
                data['genres'] += ",Software"
            else:
                data['genres'] = "Software"

    # Mature Content
    data['mature_content'] = 1 if "Adult Only" in html_content else 0
    
    # Image / Media
    header_match = re.search(r'<img class="game_header_image_full" [^>]*src="([^"]*)"', html_content)
    data['header_image'] = header_match.group(1) if header_match else ""

    return data

from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_appid(app_id):
    """
    Process a single app_id: storefront, stats, and reviews.
    Returns (game_data, reviews_list) or None if storefront is missing.
    """
    # 1. Process Storefront
    html_path = os.path.join(RAW_DOWNLOAD_PATH, f"{app_id}_storefront.html")
    if not os.path.exists(html_path):
        return None
        
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            game_data = parse_storefront_html(app_id, f.read())
    except Exception as e:
        return None
        
    if not game_data:
        return None

    # 2. Process Review Stats
    stats_path = os.path.join(RAW_DOWNLOAD_PATH, f"{app_id}_stats_english.json")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(RAW_DOWNLOAD_PATH, f"{app_id}_stats.json")
        
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                game_data['positive'] = stats.get('total_positive', 0)
                game_data['negative'] = stats.get('total_negative', 0)
                game_data['user_score'] = stats.get('review_score', 0)
                game_data['recommendations'] = stats.get('total_reviews', 0)
        except:
            pass
    
    # Defaults
    game_data.setdefault('positive', 0)
    game_data.setdefault('negative', 0)
    game_data.setdefault('user_score', 0)
    game_data.setdefault('recommendations', 0)
    game_data['owners'] = "0 .. 20,000"
    game_data['average_forever'] = 0
    game_data['median_forever'] = 0
    game_data['is_dlc'] = 'Downloadable Content' in game_data.get('categories', '')
    game_data['movies'] = ""
    game_data['required_age'] = 0
    
    # 3. Process Reviews
    app_reviews = []
    for page in range(10):
        rev_path = os.path.join(RAW_DOWNLOAD_REVIEWS_PATH, f"{app_id}_reviews_p{page}.json")
        if not os.path.exists(rev_path):
            break
        try:
            with open(rev_path, 'r', encoding='utf-8') as f:
                rev_data = json.load(f)
                for r in rev_data.get('reviews', []):
                    rev_text = r.get('review', '')
                    if is_english(rev_text):
                        author = r.get('author', {})
                        app_reviews.append({
                            'appid': app_id,
                            'review_id': r.get('recommendationid'),
                            'review_text': clean_text(rev_text),
                            'timestamp_created': r.get('timestamp_created'),
                            'voted_up': r.get('voted_up'),
                            'author_playtime_forever': author.get('playtime_forever', 0)
                        })
        except:
            break
            
    return game_data, app_reviews

def build_dataset(output_file="scraped_games.csv", reviews_file="scraped_reviews.csv"):
    """
    Step 2: Parse cached raw files into CSVs using parallel processing.
    """
    if not os.path.exists(RAW_DOWNLOAD_PATH):
        logger.error(f"Raw download path {RAW_DOWNLOAD_PATH} does not exist.")
        return

    # 1. Find all appids in cache
    app_ids = set()
    for f in os.listdir(RAW_DOWNLOAD_PATH):
        if f.endswith("_storefront.html"):
            app_ids.add(f.split("_")[0])
    
    app_ids = sorted(list(app_ids)) # Sort for deterministic progress
    logger.info(f"Found {len(app_ids)} cached games to process using multiprocessing.")

    all_games = []
    all_reviews = []
    
    # Use max available cores
    max_workers = os.cpu_count() or 4
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_appid = {executor.submit(process_single_appid, aid): aid for aid in app_ids}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_appid), total=len(app_ids), desc="Building dataset", smoothing=0):
            result = future.result()
            if result:
                game_data, app_reviews = result
                all_games.append(game_data)
                all_reviews.extend(app_reviews)

    # Save to CSV
    if all_games:
        pd.DataFrame(all_games).to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
        logger.info(f"Saved {len(all_games)} games to {output_file}")
    
    if all_reviews:
        pd.DataFrame(all_reviews).to_csv(reviews_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
        logger.info(f"Saved {len(all_reviews)} reviews to {reviews_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Build CSV dataset from raw cache.")
    parser.add_argument("--output", default="scraped_games.csv", help="Output games CSV.")
    parser.add_argument("--reviews_output", default="scraped_reviews.csv", help="Output reviews CSV.")
    
    args = parser.parse_args()
    build_dataset(output_file=args.output, reviews_file=args.reviews_output)
