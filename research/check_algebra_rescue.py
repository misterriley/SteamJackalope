import pandas as pd
import numpy as np
import ast
import json
import re
import os
from common.utils import calculate_jackalope_kernel
from common.constants import (
    METADATA_FILE, DIFFUSED_VERB_PROFILES_FILE,
    EMBEDDINGS_DESC_FILE, EMBEDDINGS_DESC_NORMS_FILE,
    TOPIC_DISTRIBUTIONS_FILE, PRODUCTION_DATA_DIR,
    TAG_GLOBAL_SCALING_FACTOR, DOT_PRODUCT_LAMBDA,
    SEMANTIC_GLOBAL_SCALING_FACTOR, SEMANTIC_DOT_PRODUCT_LAMBDA
)

df = pd.read_parquet(METADATA_FILE)
seed_appid = 1194840 # Frog Fractions
target_appid = 1379510 # Algebra Ridge

match = df[df['appid'] == seed_appid]
tags_s = match.iloc[0]['tags']
if isinstance(tags_s, str): tags_s = ast.literal_eval(tags_s)

HIGH_VALUE_NOUNS = {'Education', 'Math', 'Comedy', 'Surreal', 'Typing', 'Spelling', 'Mystery', 'Word Game'}
shared_hv_nouns = set(tags_s.keys()) & HIGH_VALUE_NOUNS
print(f"Shared HV Nouns: {shared_hv_nouns}")

t_match = df[df['appid'] == target_appid]
t_idx = t_match.index[0]
t_tags = t_match.iloc[0]['tags']
if isinstance(t_tags, str): t_tags = ast.literal_eval(t_tags)

count = len(set(t_tags.keys()) & shared_hv_nouns)
print(f"Algebra Ridge (15493) Shared Count: {count}")

# Check shared_counts calculation logic
def count_hv_matches(t_tags_s):
    if isinstance(t_tags_s, str): t_tags_s = ast.literal_eval(t_tags_s)
    return len(set(t_tags_s.keys()) & shared_hv_nouns)

shared_counts = df['tags'].apply(count_hv_matches).values
print(f"Calculated shared_counts[15493]: {shared_counts[15493]}")
