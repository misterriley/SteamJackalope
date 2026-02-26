import re
import struct

def find_bson_start(data, anchor_str):
    pos = data.find(anchor_str.encode())
    if pos == -1:
        print("Anchor not found")
        return None
    
    print("Anchor found at", pos)
    snippet = data[max(0, pos-100):pos+200]
    print("Hex around anchor:")
    print(snippet.hex())
    
    return pos

if __name__ == "__main__":
    with open('research/games.db', 'rb') as f:
        data = f.read(100000) 
        find_bson_start(data, "Description")
