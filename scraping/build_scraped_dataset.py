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
    price_match = re.search(r'<div class="game_purchase_price price" [^>]*>([^<]*)</div>', html_content)
    if not price_match:
        price_match = re.search(r'<div class="discount_final_price">([^<]*)</div>', html_content)
    data['price'] = clean_text(price_match.group(1)) if price_match else ""

    # 7. Dev/Pub
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

    # 10. Categories
    categories = re.findall(r'<a class="name" href="https://store.steampowered.com/search/\?(?:category2|category3)=\d+">([^<]*)</a>', html_content)
    data['categories'] = ",".join(set(categories))

    # Mature Content
    data['mature_content'] = 1 if "Adult Only" in html_content else 0
    
    # Image / Media
    header_match = re.search(r'<img class="game_header_image_full" [^>]*src="([^"]*)"', html_content)
    data['header_image'] = header_match.group(1) if header_match else ""

    return data

def build_dataset(output_file="scraped_games.csv", reviews_file="scraped_reviews.csv"):
    """
    Step 2: Parse cached raw files into CSVs.
    """
    if not os.path.exists(RAW_DOWNLOAD_PATH):
        logger.error(f"Raw download path {RAW_DOWNLOAD_PATH} does not exist.")
        return

    # 1. Find all appids in cache
    app_ids = set()
    for f in os.listdir(RAW_DOWNLOAD_PATH):
        if f.endswith("_storefront.html"):
            app_ids.add(f.split("_")[0])
    
    logger.info(f"Found {len(app_ids)} cached games to process.")

    all_games = []
    all_reviews = []
    
    for app_id in tqdm(app_ids, desc="Building dataset"):
        # Process Storefront
        html_path = os.path.join(RAW_DOWNLOAD_PATH, f"{app_id}_storefront.html")
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                game_data = parse_storefront_html(app_id, f.read())
        except Exception as e:
            logger.warning(f"Error parsing {app_id} storefront: {e}")
            game_data = None
            
        if not game_data:
            continue

        # Process Review Stats
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
        
        all_games.append(game_data)
        
        # Process Reviews
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
                            all_reviews.append({
                                'appid': app_id,
                                'review_id': r.get('recommendationid'),
                                'review_text': clean_text(rev_text),
                                'timestamp_created': r.get('timestamp_created'),
                                'voted_up': r.get('voted_up')
                            })
            except:
                break

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
