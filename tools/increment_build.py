import json
import os

def increment_build():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    build_count_file = os.path.join(root_dir, "build_count.json")
    
    if os.path.exists(build_count_file):
        with open(build_count_file, "r") as f:
            data = json.load(f)
    else:
        data = {"build_count": 0}
        
    data["build_count"] += 1
    
    with open(build_count_file, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Build count incremented to {data['build_count']}")

if __name__ == "__main__":
    increment_build()
