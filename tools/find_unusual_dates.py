import pandas as pd
import numpy as np
import re

def is_nice_date(s):
    if pd.isna(s) or s == "":
        return True
    
    s = str(s).strip()
    
    # Match "YYYY-MM-DD"
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return True
    
    # Match "YYYY"
    if re.match(r'^\d{4}$', s):
        return True
    
    # Match "Month YYYY" or "YYYY Month"
    if re.match(r'^([A-Za-z]+)\s+(\d{4})$', s):
        return True
    if re.match(r'^(\d{4})\s+([A-Za-z]+)$', s):
        return True

    # Try standard pandas parsing
    try:
        pd.to_datetime(s, errors='raise')
        return True
    except:
        return False

print("Reading scraped_games.csv...")
df = pd.read_csv('scraped_games.csv', usecols=['appid', 'name', 'release_date'])

print("Finding unusual release dates...")
# Filter for things that don't parse nicely
unusual_mask = df['release_date'].apply(lambda x: not is_nice_date(x))
unusual_dates = df[unusual_mask]['release_date'].unique()

print("\n--- Unique Unusual Release Dates Found ---")
for d in sorted([str(x) for x in unusual_dates]):
    print(f"'{d}'")

# Also show some examples of names for these dates
print("\n--- Examples ---")
print(df[unusual_mask][['name', 'release_date']].head(20))
