import re
import struct

def find_factorio_name(db_path):
    with open(db_path, 'rb') as f:
        data = f.read()
    
    # Search for any string field containing "Factorio"
    pattern = b'\x02' # String type
    matches = re.finditer(pattern, data)
    
    for m in matches:
        pos = m.start()
        # Potential string field: [0x02] [Name\0] [Len] [Val\0]
        # We don't know the field name, so we look for the value
        if b'Factorio' in data[pos:pos+100]:
            # Try to find the field name before it
            # Field name is usually alphanumeric
            field_name_match = re.search(b'([a-zA-Z0-9]+)\x00', data[max(0, pos-50):pos])
            if field_name_match:
                fname = field_name_match.group(1).decode('utf-8', 'ignore')
                # Try to extract the string value
                try:
                    # The value starts after the field name and null byte
                    val_pos = data.find(b'\x00', pos + 1) + 1
                    length = struct.unpack('<i', data[val_pos:val_pos+4])[0]
                    val = data[val_pos+4:val_pos+4+length-1].decode('utf-8', 'ignore')
                    if 'Factorio' in val:
                        print(f"Found field '{fname}' with value '{val}' at {pos}")
                except:
                    pass

if __name__ == "__main__":
    find_factorio_name('research/games.db')
