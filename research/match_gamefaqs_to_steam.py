import pandas as pd
from bs4 import BeautifulSoup
import os
import re
import difflib

# Roman to Arabic numeral mapping
ROMAN_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20
}

# Word to Arabic numeral mapping
WORD_MAP = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

def normalize_title(title):
    if not isinstance(title, str):
        return ""
    
    # Lowercase
    title = title.lower()
    
    # Handle specific editions/suffixes that might confuse matching
    title = re.sub(r'\b(edition|enhanced|goty|game of the year|complete|anniversary|remastered|remake|collection|bundle|definitive|director\'s cut|hd)\b', '', title)
    
    # Replace roman numerals at the end or surrounded by spaces
    for roman, arabic in ROMAN_MAP.items():
        title = re.sub(r'\b' + roman.lower() + r'\b', str(arabic), title)
        
    # Replace word numerals
    for word, arabic in WORD_MAP.items():
        title = re.sub(r'\b' + word + r'\b', str(arabic), title)
        
    # Remove punctuation but keep numbers and letters
    title = re.sub(r'[^\w\s]', '', title)
    
    # Normalize whitespace
    title = ' '.join(title.split())
    
    return title

def parse_all_gamefaqs_files(directory):
    all_data = []
    files = [f for f in os.listdir(directory) if f.endswith('.html')]
    print(f"Parsing {len(files)} files...")
    
    for filename in files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        if not table:
            continue
            
        tbody = table.find('tbody')
        if not tbody:
            continue
            
        rows = tbody.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                # GameFAQs structure check
                a_tag = cols[1].find('a')
                if not a_tag:
                    continue
                title = a_tag.text.strip()
                difficulty = cols[2].text.strip()
                all_data.append({'gf_title': title, 'difficulty': difficulty})
    
    df = pd.DataFrame(all_data)
    # Handle duplicates: convert difficulty to float and take mean
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')
    df = df.dropna(subset=['difficulty'])
    df = df.groupby('gf_title')['difficulty'].mean().reset_index()
    
    return df

def has_numeric_mismatch(s1, s2):
    """Checks if two strings have different numbers, suggesting a mismatch (e.g. 'game 1' vs 'game 2')."""
    nums1 = set(re.findall(r'\d+', s1))
    nums2 = set(re.findall(r'\d+', s2))
    return nums1 != nums2

def perform_matching(gf_df, steam_df):
    results = []
    
    # Pre-normalize Steam titles for faster lookup
    print("Normalizing Steam titles...")
    steam_df['norm_name'] = steam_df['name'].apply(normalize_title)
    
    # Create mapping from normalized name to appid and original name
    norm_to_steam = {}
    for _, row in steam_df.iterrows():
        norm = row['norm_name']
        if norm:
            if norm not in norm_to_steam:
                norm_to_steam[norm] = []
            norm_to_steam[norm].append((row['appid'], row['name']))
            
    print(f"Matching {len(gf_df)} GameFAQs entries...")
    
    all_steam_norms = list(norm_to_steam.keys())
    
    for idx, row in gf_df.iterrows():
        gf_title = row['gf_title']
        norm_gf = normalize_title(gf_title)
        if not norm_gf:
            continue
            
        match_found = False
        confidence = 0.0
        match_appid = None
        match_steam_name = None
        
        # 1. Exact match after normalization
        if norm_gf in norm_to_steam:
            matches = norm_to_steam[norm_gf]
            match_appid, match_steam_name = matches[0]
            confidence = 1.0 if len(matches) == 1 else 0.95
            match_found = True
        
        # 2. Substring match (conservative)
        if not match_found:
            # CHECK: Single word titles (no numbers/punctuation) should only match exactly
            # We already checked exact matches in step 1, so if it's a single pure word,
            # we skip substring/fuzzy matching.
            is_single_pure_word = re.match(r'^[a-z]+$', norm_gf)
            
            if not is_single_pure_word:
                best_ratio = 0
                best_match = None
                for steam_norm in all_steam_norms:
                    if norm_gf in steam_norm or steam_norm in norm_gf:
                        # Ignore if numeric mismatch (prevents 'Doom' matching 'Doom 2')
                        if has_numeric_mismatch(norm_gf, steam_norm):
                            continue
                            
                        ratio = min(len(norm_gf), len(steam_norm)) / max(len(norm_gf), len(steam_norm))
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = steam_norm
                
                if best_match and best_ratio > 0.7:
                    match_appid, match_steam_name = norm_to_steam[best_match][0]
                    confidence = 0.9 * best_ratio
                    match_found = True

        # 3. Fuzzy match
        if not match_found:
            is_single_pure_word = re.match(r'^[a-z]+$', norm_gf)
            if not is_single_pure_word:
                close_matches = difflib.get_close_matches(norm_gf, all_steam_norms, n=1, cutoff=0.9)
                if close_matches:
                    match_norm = close_matches[0]
                    # Still check numeric mismatch
                    if not has_numeric_mismatch(norm_gf, match_norm):
                        match_appid, match_steam_name = norm_to_steam[match_norm][0]
                        confidence = 0.85
                        match_found = True
        
        if match_found:
            results.append({
                'gf_title': gf_title,
                'gf_difficulty': row['difficulty'],
                'steam_appid': match_appid,
                'steam_name': match_steam_name,
                'confidence': round(confidence, 2)
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load data
    gf_dir = 'data/GameFAQs'
    steam_csv = 'data/pipeline_games_clean.csv'
    
    gf_df = parse_all_gamefaqs_files(gf_dir)
    steam_df = pd.read_csv(steam_csv)
    
    # Perform matching
    match_df = perform_matching(gf_df, steam_df)
    
    # Save results
    output_path = 'data/gamefaqs_to_steam_match.csv'
    match_df.to_csv(output_path, index=False)
    
    print(f"Successfully matched {len(match_df)} games out of {len(gf_df)} unique GameFAQs titles.")
    print(f"Results saved to {output_path}")
    print(match_df.head())
