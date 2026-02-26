import re
import struct
import os
import json

def parse_playnite_db(db_path, output_json=None):
    if not os.path.exists(db_path):
        print(f"File not found: {db_path}")
        return None
        
    with open(db_path, 'rb') as f:
        data = f.read()
        
    # Find all document starts
    id_pattern = b'\x05_id\x00'
    indices = [m.start() for m in re.finditer(id_pattern, data)]
    
    games = []
    
    exclude_names = {
        "steam", "official", "wikipedia", "wikia", "twitch", "youtube", 
        "facebook", "twitter", "instagram", "discord", "epic", "gog", 
        "gog.com", "uknown", "unknown", "iphone", "ipad", "android", 
        "pcgw", "reddit", "news", "forum", "store", "wiki", "bluesky", "itch", "none"
    }
    
    print(f"Scanning {len(indices)} documents in {db_path}...")
    
    for i in range(len(indices)):
        start = indices[i]
        end = indices[i+1] if i+1 < len(indices) else len(data)
        doc = data[start:end]
        
        # 1. Find Steam AppID (Primary indicator of a Steam game)
        appid_match = re.search(br'https://store.steampowered.com/app/(\d+)', doc)
        if not appid_match:
            continue
        appid = int(appid_match.group(1).decode())
        
        # 2. Find Playtime
        playtime = 0
        p_match = re.search(b'([\x10\x12])Playtime\x00', doc)
        if p_match:
            t_byte = p_match.group(1)[0]
            p_pos = p_match.end()
            try:
                if t_byte == 0x12: # Int64
                    playtime = struct.unpack('<q', doc[p_pos:p_pos+8])[0]
                elif t_byte == 0x10: # Int32
                    playtime = struct.unpack('<i', doc[p_pos:p_pos+4])[0]
            except:
                pass
        
        # Sanity check: Playnite stores playtime in seconds
        if playtime > 2 * 10**9 or playtime < 0:
            playtime = 0
            
        # 3. Heuristic Name Extraction
        game_name = "Unknown"
        
        # Try finding Name fields not in the exclude list
        names = []
        for nm in re.finditer(b'\x02Name\x00', doc):
            n_pos = nm.end()
            try:
                length = struct.unpack('<i', doc[n_pos:n_pos+4])[0]
                if 1 < length < 256:
                    name_str = doc[n_pos+4:n_pos+4+length-1].decode('utf-8', 'ignore').strip()
                    if name_str.lower() not in exclude_names:
                        names.append(name_str)
            except:
                pass
        
        if names:
            game_name = names[-1]
            
        # Fallback to Description <strong> tag
        if game_name == "Unknown":
            # Look for <strong>...</strong> in the first 500 bytes of description
            desc_pos = doc.find(b'Description\x00')
            if desc_pos != -1:
                snippet = doc[desc_pos:desc_pos+1000]
                strong_match = re.search(b'<strong>(.*?)</strong>', snippet)
                if strong_match:
                    try:
                        game_name = strong_match.group(1).decode('utf-8', 'ignore')
                    except:
                        pass
        
        games.append({
            'appid': appid,
            'name': game_name,
            'playtime_seconds': int(playtime),
            'playtime_hrs': round(playtime / 3600.0, 2)
        })
        
    # Aggregate by AppID (taking the one with most playtime/best name)
    aggregated = {}
    for g in games:
        aid = g['appid']
        if aid not in aggregated:
            aggregated[aid] = g
        else:
            # Update if we find a better name or more playtime
            if aggregated[aid]['name'] == "Unknown" and g['name'] != "Unknown":
                aggregated[aid]['name'] = g['name']
            if g['playtime_seconds'] > aggregated[aid]['playtime_seconds']:
                aggregated[aid]['playtime_seconds'] = g['playtime_seconds']
                aggregated[aid]['playtime_hrs'] = g['playtime_hrs']
            
    final_list = sorted(aggregated.values(), key=lambda x: x['playtime_seconds'], reverse=True)
    
    print(f"Found {len(final_list)} unique Steam games.")
    played = [g for g in final_list if g['playtime_seconds'] > 0]
    print(f"Games with recorded playtime: {len(played)}")
    
    print("\nTop 20 Played Games (Steam only):")
    for g in final_list[:20]:
        print(f"- {g['name']} ({g['appid']}): {g['playtime_hrs']} hrs")
        
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(final_list, f, indent=4)
        print(f"\nResults saved to {output_json}")
        
    return final_list

if __name__ == "__main__":
    parse_playnite_db('research/games.db', 'playnite_steam_games.json')
