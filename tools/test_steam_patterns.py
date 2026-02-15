import requests
import urllib.parse

def test_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, allow_redirects=True)
    print(f"Testing: {url}")
    print(f"  Status: {response.status_code}")
    print(f"  Final URL: {response.url}")
    
    final_url = response.url.split('?')[0].rstrip('/')
    home_page = "https://store.steampowered.com"
    is_home = final_url == home_page or final_url == home_page + "/search"
    print(f"  Valid: {not is_home}")
    print("-" * 20)

terms = ["Action", "Indie", "Single-player", "1990's", "Zombies"]
patterns = [
    "https://store.steampowered.com/tags/en/{name}",
    "https://store.steampowered.com/genre/{name}",
    "https://store.steampowered.com/category/{name}"
]

for term in terms:
    print(f"TERM: {term}")
    test_url(patterns[0].format(name=urllib.parse.quote(term)))
    test_url(patterns[1].format(name=urllib.parse.quote(term)))
    test_url(patterns[2].format(name=urllib.parse.quote(term)))
    print("=" * 40)
