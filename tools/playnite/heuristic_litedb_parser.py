import re
import struct
import os

def extract_game_blobs(db_path):
    with open(db_path, 'rb') as f:
        data = f.read()
        
    # Search for _id field which marks the start of a document in LiteDB v5
    # Playnite uses binary GUIDs for _id
    id_pattern = b'\x05_id\x00'
    
    indices = [m.start() for m in re.finditer(id_pattern, data)]
    print(f"Found {len(indices)} potential documents")
    
    results = []
    
    for i in range(len(indices)):
        start = indices[i]
        end = indices[i+1] if i+1 < len(indices) else len(data)
        blob = data[start:end]
        
        # Check if it's a Steam game
        link_match = re.search(br'https://store.steampowered.com/app/(\d+)', blob)
        if link_match:
            appid = int(link_match.group(1).decode())
            
            # Extract Playtime
            # Playtime\x00 [type] [value]
            playtime_match = re.search(b'Playtime\x00', blob)
            playtime_mins = 0
            if playtime_match:
                p_pos = playtime_match.end()
                type_byte = blob[p_pos]
                try:
                    if type_byte == 0x12:
                        playtime_seconds = struct.unpack('<q', blob[p_pos+1:p_pos+9])[0]
                    elif type_byte == 0x10:
                        playtime_seconds = struct.unpack('<i', blob[p_pos+1:p_pos+5])[0]
                    else:
                        playtime_seconds = 0
                    playtime_mins = playtime_seconds // 60
                except:
                    pass
            
            # Extract Name
            name_match = re.search(b'Name\x00\x02', blob)
            name = "Unknown"
            if name_match:
                n_pos = name_match.end()
                length = struct.unpack('<i', blob[n_pos:n_pos+4])[0]
                name = blob[n_pos+4:n_pos+4+length-1].decode('utf-8', 'ignore')
                
            results.append({
                'appid': appid,
                'name': name,
                'playtime': playtime_mins
            })
            
    print(f"Extracted {len(results)} Steam games")
    played = [r for r in results if r['playtime'] > 0]
    print(f"Games with playtime: {len(played)}")
    
    for r in sorted(played, key=lambda x: x['playtime'], reverse=True)[:20]:
        print(f"{r['name']} ({r['appid']}): {r['playtime']} mins")

if __name__ == "__main__":
    extract_game_blobs('research/games.db')
