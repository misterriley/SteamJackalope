import re
import os

path = 'C:/steam_raw_downloads/848450_storefront.html'
with open(path, 'rb') as f:
    raw = f.read()

content = raw.decode('utf-8', errors='ignore')
print(f"File size: {len(content)}")

# Look for index of "ss_"
idx = content.find("ss_")
if idx != -1:
    print(f"Found 'ss_' at index {idx}")
    print("Context:")
    print(content[idx-100:idx+300])
else:
    print("'ss_' not found in file")

# Try very simple regex
simple = re.findall(r'ss_[a-f0-9]+\.jpg', content)
print(f"Simple matches: {len(simple)}")
if simple:
    print(f"First simple match: {simple[0]}")
