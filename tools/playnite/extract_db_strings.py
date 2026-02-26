import re
import os

def extract_strings(file_path, limit=100):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, 'rb') as f:
        # Read a good chunk of the file
        data = f.read(100000) 
        # Find sequences of printable characters
        strings = re.findall(b'[\x20-\x7E]{4,}', data)
        
        print(f"Extracted first {limit} strings from {file_path}:")
        for s in strings[:limit]:
            try:
                decoded = s.decode('utf-8')
                # Filter out some noise
                if len(decoded) > 5 and not decoded.startswith('***'):
                    print(decoded)
            except:
                pass

if __name__ == "__main__":
    extract_strings('research/games.db')
