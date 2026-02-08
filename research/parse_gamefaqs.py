import pandas as pd
from bs4 import BeautifulSoup
import os

def parse_gamefaqs_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    
    if not table:
        print("No table found in HTML.")
        return []
    
    rows = table.find('tbody').find_all('tr')
    data = []
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:
            platform = cols[0].text.strip()
            title = cols[1].find('a').text.strip()
            difficulty = cols[2].text.strip()
            data.append({
                'title': title,
                'difficulty': difficulty,
                'platform': platform
            })
            
    return data

if __name__ == "__main__":
    html_file = 'data/GameFAQs/Game Search - GameFAQs (2_5_2026 3：49：37 PM).html'
    results = parse_gamefaqs_html(html_file)
    
    if results:
        df = pd.DataFrame(results)
        output_file = 'research/difficulty_ratings_gamefaqs.csv'
        df.to_csv(output_file, index=False)
        print(f"Successfully extracted {len(results)} games to {output_file}")
        print(df.head())
    else:
        print("Failed to extract data.")
