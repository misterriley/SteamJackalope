import re
import struct

def dump_blob_fields(db_path, start_pos):
    with open(db_path, 'rb') as f:
        f.seek(start_pos)
        data = f.read(5000)
    
    end = data.find(b'\x05_id\x00', 1)
    if end != -1:
        data = data[:end]
        
    print(f"Dumping fields for blob at {start_pos}:")
    # Matches [Type] [Name\0]
    matches = re.finditer(b'([\x01-\x1F])([a-zA-Z0-9]{3,})\x00', data)
    for m in matches:
        t_byte = m.group(1)[0]
        f_name = m.group(2).decode()
        print(f"Type: {hex(t_byte)}, Field: {f_name}")
        
        # If it's a string (0x02) or other common type, try to show value
        val_start = m.end()
        if t_byte == 0x02:
            try:
                length = struct.unpack('<i', data[val_start:val_start+4])[0]
                val = data[val_start+4:val_start+4+length-1].decode('utf-8', 'ignore')
                print(f"  String: {val[:100]}")
            except:
                pass
        elif t_byte == 0x12:
            try:
                val = struct.unpack('<q', data[val_start:val_start+8])[0]
                print(f"  Int64: {val}")
            except:
                pass

if __name__ == "__main__":
    dump_blob_fields('research/games.db', 14770205)
