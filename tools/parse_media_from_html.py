import os
import re
import json
import html
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_DIR = "C:/steam_raw_downloads"
CACHE_DIR = "data/media_cache"

def parse_media(html_content, appid):
    screenshots = []
    movies = [] # Now stores {url, poster}
    
    # Method 1: Data props (Structured)
    carousel_match = re.search(r'data-featuretarget="gamehighlight-desktopcarousel"\s+data-props="(.*?)"', html_content)
    if carousel_match:
        try:
            props_str = html.unescape(carousel_match.group(1))
            # Critical: Handle escaped slashes which html.unescape misses
            props_str = props_str.replace('\\/', '/')
            props = json.loads(props_str)
            
            # Extract trailers
            for trailer in props.get('trailers', []):
                movie_data = {"url": None, "poster": None}
                
                # Get poster
                movie_data["poster"] = trailer.get('poster') or trailer.get('thumbnail')
                
                # Strategy 1: Extract Video ID from thumbnail (Most reliable)
                thumbnail = trailer.get('thumbnail', '')
                video_id_match = re.search(r'/apps/(\d+)/', thumbnail)
                if video_id_match:
                    vid = video_id_match.group(1)
                    movie_data["url"] = f"https://cdn.akamai.steamstatic.com/steam/apps/{vid}/movie_max.mp4"
                
                # Strategy 2: Use trailer ID (Fallback)
                if not movie_data["url"]:
                    trailer_id = trailer.get('id')
                    if trailer_id:
                        movie_data["url"] = f"https://video.akamai.steamstatic.com/store_trailers/{appid}/{trailer_id}/movie_max.mp4"
                
                # Strategy 3: Direct mp4 from props
                if not movie_data["url"]:
                    mp4_data = trailer.get('mp4', {})
                    movie_data["url"] = mp4_data.get('max') or mp4_data.get('480')
                
                if movie_data["url"]:
                    movies.append(movie_data)
            
            # Extract screenshots
            for ss in props.get('screenshots', []):
                if ss.get('full'):
                    screenshots.append(ss['full'])
                elif ss.get('standard'):
                    screenshots.append(ss['standard'])
        except Exception:
            pass

    # Method 2: Resilient Screenshot Parsing
    if not screenshots:
        clean_content = html_content.replace('\\', '')
        ss_clean = re.findall(r'(https://[^\s"\'<>&]*?ss_[^\s"\'<>&]*?\.(?:jpg|png))', clean_content)
        screenshots.extend(ss_clean)
    
    # Deduplicate screenshots by ID, picking highest resolution
    ss_dict = {}
    for url in screenshots:
        match = re.search(r'(ss_[a-f0-9]+)', url)
        if not match: continue
        ss_id = match.group(1)
        score = 0
        if '1920x1080' in url: score = 3
        elif '600x338' in url: score = 2
        elif url.endswith('.jpg') or url.endswith('.png'): score = 2.5
        if ss_id not in ss_dict or score > ss_dict[ss_id][0]:
            ss_dict[ss_id] = (score, url)
    screenshots = [val[1] for val in ss_dict.values()]
        
    # Fallback for movies if structured parsing failed
    if not movies:
        video_matches = re.findall(r'<source src="(https://shared\.(?:akamai|fastly)\.steamstatic\.com/store_item_assets/steam/apps/\d+/extras/[^?"]*\.(?:mp4|webm))"', html_content)
        for v in list(set(video_matches)):
            movies.append({"url": v, "poster": None})

    # Deduplicate movies by URL
    seen_movies = set()
    final_movies = []
    for m in movies:
        if m["url"] not in seen_movies:
            seen_movies.add(m["url"])
            final_movies.append(m)

    return {
        "screenshots": screenshots,
        "movies": final_movies
    }

def process_file(filename):
    appid = filename.split("_")[0]
    cache_path = os.path.join(CACHE_DIR, f"{appid}.json")
    
    try:
        with open(os.path.join(RAW_DIR, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            
        media = parse_media(content, appid)
        
        if media['screenshots'] or media['movies']:
            with open(cache_path, 'w') as f:
                json.dump(media, f)
            return True
    except Exception:
        pass
    return False

def main():
    if not os.path.exists(RAW_DIR):
        print(f"Error: {RAW_DIR} not found.")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith("_storefront.html")]
    max_workers = os.cpu_count() or 4
    print(f"Parsing {len(files)} files with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Parsing Media"):
            future.result()
    print(f"Finished updating media cache.")

if __name__ == "__main__":
    main()
