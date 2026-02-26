import re
import struct
import os

def scan_playtimes(db_path):
    with open(db_path, 'rb') as f:
        data = f.read()
        
    # Search for [Type 0x12] [Playtime \0]
    pattern = b'\x12Playtime\x00'
    matches = re.finditer(pattern, data)
    
    found = 0
    for m in matches:
        pos = m.end()
        try:
            val = struct.unpack('<q', data[pos:pos+8])[0]
            if 0 < val < 10**10: # Reasonable range (up to 300 years)
                print(f"Found Playtime at {m.start()}: {val} seconds ({val/3600.0:.1f} hours)")
                found += 1
        except:
            pass
    print(f"Found {found} reasonable playtimes")

if __name__ == "__main__":
    scan_playtimes('research/games.db')
