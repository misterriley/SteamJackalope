import json
import os
from common.constants import REGULARIZATION_FILE
if os.path.exists(REGULARIZATION_FILE):
    with open(REGULARIZATION_FILE, 'r') as f:
        data = json.load(f)
        print(f"MEAN: {data.get('SEMANTIC_SIMILARITY_MEAN')}")
        print(f"STD:  {data.get('SEMANTIC_SIMILARITY_STD')}")
else:
    print("Not found")
