
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer # Moved to lazy load
import os
import sys
import re
import ast
import gc

# Memory Optimization: Limit threads to reduce buffer overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add parent directory to sys.path so we can import common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.constants import (
    EMBEDDINGS_DESC_FILE,
    EMBEDDINGS_TAG_FILE,
    QUALITY_GRID_FILE,
    W_DESC_FILE,
    W_STRUCTURAL_FILE,
    MEAN_DESC_FILE,
    MEAN_STRUCTURAL_FILE,
    TAG_VECTORS_FILE,
    TAG_NORMS_FILE,
    METADATA_FILE,
    DOT_PRODUCT_LAMBDA,
    EPSILON,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX,
    TOP_K_SORT_MULTIPLIER,
    SEMANTIC_PROMPT_SEED_BLEND,
    MODEL_NAME,
    AP_SLIDER_VALUES,
    AP_SLIDER_STEP,
    AP_SLIDER_MIN,
    SEMANTIC_WEIGHT_MULTIPLIER,
    TAG_WEIGHT_MULTIPLIER,
    QUALITY_WEIGHT_MULTIPLIER,
    AGE_WEIGHT_MULTIPLIER,
    POPULARITY_WEIGHT_MULTIPLIER,
    LENGTH_WEIGHT_MULTIPLIER,
    DIFFICULTY_WEIGHT_MULTIPLIER,
    NSFW_TAGS,
    NSFW_NAME_PATTERNS
)
from common.utils import to_z, calculate_hybrid_score

app = FastAPI()

# --- Data Loading ---

class DataManager:
    def __init__(self):
        self.embeddings_desc_norm = None
        self.embeddings_structural_norm = None
        self.metadata = None
        self.tag_vectors = None
        self.quality_grid = None
        self.tag_vectors_norms = None
        self.w_desc = None
        self.w_structural = None
        self.mean_desc = None
        self.mean_structural = None
        self._model = None
        self.all_genres = []

    @property
    def model(self):
        if self._model is None:
            print("Loading SentenceTransformer model (Lazy Load)...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
            
            # Attempt dynamic quantization to reduce memory usage (Linux/CPU optimization)
            # Update: Quantization increases startup memory overhead significantly in PyTorch.
            # Disabling it might actually keep peak memory lower (around 350MB vs 475MB).
            # try:
            #     import torch
            #     torch.set_num_threads(1) # Explicitly set torch threads
            #     print("Quantizing model to qint8...")
            #     self._model = torch.quantization.quantize_dynamic(
            #         self._model, {torch.nn.Linear}, dtype=torch.qint8
            #     )
            #     print("Model quantized.")
            #     # Force garbage collection to free original float32 weights
            #     gc.collect()
            # except Exception as e:
            #     print(f"Quantization failed: {e}")
            pass
                
        return self._model

    def clean_release_date(self, date_str):
        if pd.isna(date_str) or date_str == "":
            return pd.NaT
        
        s = str(date_str).strip()
        
        # Match "YYYY-MM-DD"
        if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
            return pd.to_datetime(s, errors='coerce')
        
        # Match "YYYY" -> YYYY-07-01
        if re.match(r'^\d{4}$', s):
            return pd.to_datetime(s + "-07-01", errors='coerce')
        
        # Match "Month YYYY" or "YYYY Month" (assume 15th)
        m1 = re.match(r'^([A-Za-z]+)\s+(\d{4})$', s)
        if m1:
            return pd.to_datetime(f"{m1.group(1)} 15, {m1.group(2)}", errors='coerce')
        
        m2 = re.match(r'^(\d{4})\s+([A-Za-z]+)$', s)
        if m2:
            return pd.to_datetime(f"{m2.group(2)} 15, {m2.group(1)}", errors='coerce')

        # Default fallback for "DD Mon, YYYY" etc.
        return pd.to_datetime(s, errors='coerce')

    def load_data(self):
        print("Loading data...")
        
        # 1. Use Memory Mapping for large NumPy arrays
        # Note: mmap arrays are read-only to prevent accidentally modifying on-disk data
        # We also cast to float16 to reduce memory when we DO load into RAM
        
        # 1. Use Memory Mapping for large NumPy arrays
        # Note: mmap arrays are read-only to prevent accidentally modifying on-disk data
        # Embeddings are pre-normalized by the pipeline/tools to allow for direct mmap usage
        self.embeddings_desc_norm = np.load(EMBEDDINGS_DESC_FILE, mmap_mode='r')
        self.embeddings_structural_norm = np.load(EMBEDDINGS_TAG_FILE, mmap_mode='r')

        # 2. Memory Map Tag Vectors and Quality Grid
        # These are large and used for indexing/dot products, mmap is perfect here
        self.tag_vectors = np.load(TAG_VECTORS_FILE, mmap_mode='r')
        self.quality_grid = np.load(QUALITY_GRID_FILE, mmap_mode='r')
        
        # 3. Load Metadata with specific columns and types
        print(f"DEBUG: METADATA_FILE is {METADATA_FILE}")
        needed_cols = [
            'appid', 'name', 'release_date', 'positive', 'negative', 
            'genres', 'tags', 'categories', 'supported_languages',
            'mature_content', 'date_z', 'pop_z', 'playtime_z', 'difficulty_z',
            'estimated_playtime', 'difficulty_predicted'
        ]
        if 'release_year' in pd.read_parquet(METADATA_FILE, columns=[]).columns:
            needed_cols.append('release_year')

        # Use pyarrow backend for memory efficiency with strings
        try:
             self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols, dtype_backend='pyarrow')
        except Exception as e:
             print(f"WARNING: Failed to load with pyarrow backend: {e}. Falling back to standard.")
             self.metadata = pd.read_parquet(METADATA_FILE, columns=needed_cols)
        
        # Extract necessary info from large string columns then drop them to save memory
        # Note: With pyarrow strings, operations are fast.
        self.metadata['is_vr_only'] = self.metadata['categories'].fillna('').astype(str).str.contains('VR Only', case=False).astype(bool)
        self.metadata['is_english'] = self.metadata['supported_languages'].fillna('').astype(str).str.contains('English', case=False).astype(bool)
        self.metadata['is_utility'] = self.metadata['tags'].fillna('').astype(str).str.contains('Utilities', case=False).astype(bool)
        self.metadata.drop(columns=['categories', 'supported_languages'], inplace=True)
        
        # Optimize metadata types
        self.metadata['appid'] = self.metadata['appid'].astype(np.int32)
        self.metadata['positive'] = self.metadata['positive'].astype(np.int32)
        self.metadata['negative'] = self.metadata['negative'].astype(np.int32)
        self.metadata['date_z'] = self.metadata['date_z'].astype(np.float16)
        self.metadata['pop_z'] = self.metadata['pop_z'].astype(np.float16)
        self.metadata['playtime_z'] = self.metadata['playtime_z'].astype(np.float16)
        self.metadata['difficulty_z'] = self.metadata['difficulty_z'].astype(np.float16)
        self.metadata['estimated_playtime'] = self.metadata['estimated_playtime'].astype(np.float32)
        self.metadata['difficulty_predicted'] = self.metadata['difficulty_predicted'].astype(np.float32)
        self.metadata['mature_content'] = self.metadata['mature_content'].fillna(0).astype(np.int8)

        # Parse date and convert string columns to categorical to save memory while keeping functionality
        # We need standard pandas datetime for dt accessor, pyarrow timestamp is tricky with some pandas ops
        self.metadata['parsed_date'] = self.metadata['release_date'].apply(self.clean_release_date)
        
        print(f"DEBUG: Metadata loaded. Rows: {len(self.metadata)}")
        
        # 4. Load weights and means
        self.w_desc = np.load(W_DESC_FILE).astype(np.float16) if os.path.exists(W_DESC_FILE) else None
        self.w_structural = np.load(W_STRUCTURAL_FILE).astype(np.float16) if os.path.exists(W_STRUCTURAL_FILE) else None
        self.mean_desc = np.load(MEAN_DESC_FILE).astype(np.float16) if os.path.exists(MEAN_DESC_FILE) else None
        self.mean_structural = np.load(MEAN_STRUCTURAL_FILE).astype(np.float16) if os.path.exists(MEAN_STRUCTURAL_FILE) else None

        if 'release_year' not in self.metadata.columns:
            self.metadata['release_year'] = self.metadata['parsed_date'].dt.year
            mean_year = self.metadata['release_year'].mean()
            self.metadata['release_year'] = self.metadata['release_year'].fillna(mean_year).astype(np.int16)
        else:
            self.metadata['release_year'] = self.metadata['release_year'].fillna(0).astype(np.int16)
        
        # Load pre-calculated tag vector norms if available to save massive RAM spike
        if os.path.exists(TAG_NORMS_FILE):
             print(f"Loading pre-calculated tag norms from {TAG_NORMS_FILE}...")
             self.tag_vectors_norms = np.load(TAG_NORMS_FILE)
        else:
             print("WARNING: Pre-calculated tag norms not found. Computing on the fly (High RAM Usage)...")
             # Use float32 for computation but avoid creating a full copy if possible? 
             # No, mmap astype creates copy. We stick to old way if file missing.
             self.tag_vectors_norms = np.linalg.norm(self.tag_vectors.astype(np.float32), axis=1).astype(np.float16)
        
        print("Processing genres...")
        # Avoid creating a list column (genres_list) to save ~100MB RAM.
        # We extract unique genres once for the filter list.
        def parse_genres_safe(x):
            if pd.isna(x): return []
            x = str(x)
            if x.startswith('[') and x.endswith(']'):
                try:
                    return ast.literal_eval(x)
                except:
                    pass
            return [g.strip() for g in x.split(',') if g.strip()]

        all_genres_set = set()
        # Process in chunks or iterate to avoid creating a massive series
        for g_str in self.metadata['genres'].dropna().unique():
            all_genres_set.update(parse_genres_safe(g_str))
        self.all_genres = sorted(list(all_genres_set))

        # print("Loading model...")
        # self.model = SentenceTransformer(MODEL_NAME)
        # Lazy load model only when needed to save startup memory
        
        mem_usage = self.metadata.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Metadata loaded. Size in RAM: {mem_usage:.2f} MB")
        print("Data loaded.")

data_manager = DataManager()

@app.on_event("startup")
async def startup_event():
    data_manager.load_data()

# --- API Models ---

class MetadataRequest(BaseModel):
    names: List[str]

class RecommendationRequest(BaseModel):
    alpha: float
    beta: float
    quality_pref: float
    age_pref: float
    pop_pref: float
    disc_pref: float
    length_pref: float
    difficulty_pref: float
    remove_vr: bool
    english_only: bool
    remove_nsfw: bool
    remove_utilities: bool
    remove_unreleased: bool
    top_k: int
    prompt: str
    seed_games: List[str]
    genres: Optional[List[str]] = []

# --- Endpoints ---

@app.get("/genres")
def get_genres():
    """Returns a list of all unique genres available in the dataset."""
    return data_manager.all_genres

@app.get("/games")
def get_games():
    """Returns a list of all available game names for the seed selector."""
    if data_manager.metadata is None:
         raise HTTPException(status_code=503, detail="Data not loaded yet")
    return sorted(data_manager.metadata['name'].fillna("Unknown").tolist())

@app.get("/lists/{category}")
def get_list(category: str, discovery_pref: float = 0.0):
    """
    Returns top/bottom game lists for various categories (quality, length, popularity, age, difficulty).
    Used by the 'Lists' tab in the frontend.
    """
    if data_manager.metadata is None:
         raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    metadata = data_manager.metadata
    
    if category == "quality":
        num_grid_rows = data_manager.quality_grid.shape[0]
        grid_index = int(round(((discovery_pref - (-1.0)) / 2.0) * (num_grid_rows - 1)))
        grid_index = max(0, min(num_grid_rows - 1, grid_index))
        scores = data_manager.quality_grid[grid_index]
        
        top_indices = np.argsort(-scores)[:50]
        bottom_indices = np.argsort(scores)[:50]
        
        def format_quality_list(indices):
            df = metadata.iloc[indices].copy()
            df['quality_score'] = scores[indices]
            return df[['appid', 'name', 'quality_score']].to_dict(orient='records')

        return {
            "top": format_quality_list(top_indices),
            "bottom": format_quality_list(bottom_indices)
        }

    elif category == "length":
        playtime_col = 'estimated_playtime'
        valid = metadata[metadata[playtime_col] > 0].copy()
        
        longest = valid.sort_values(playtime_col, ascending=False).head(50)
        # Debugging Dota 2 visibility
        if not longest.empty:
            print(f"DEBUG: Longest in 'length': {longest.iloc[0]['name']} with {longest.iloc[0][playtime_col]} minutes")
        shortest = valid.sort_values(playtime_col, ascending=True).head(50)
        
        return {
            "top": longest[['appid', 'name', playtime_col]].rename(columns={playtime_col: 'playtime'}).to_dict(orient='records'),
            "bottom": shortest[['appid', 'name', playtime_col]].rename(columns={playtime_col: 'playtime'}).to_dict(orient='records')
        }

    elif category == "popularity":
        if 'total_reviews' not in metadata.columns:
            metadata = metadata.copy()
            metadata['total_reviews'] = metadata['positive'] + metadata['negative']
        
        popular_df = metadata[metadata['total_reviews'] >= 1]
        most_pop = popular_df.sort_values('total_reviews', ascending=False).head(50)
        least_pop = popular_df.sort_values('total_reviews', ascending=True).head(50)
        
        return {
            "top": most_pop[['appid', 'name', 'total_reviews']].to_dict(orient='records'),
            "bottom": least_pop[['appid', 'name', 'total_reviews']].to_dict(orient='records')
        }

    elif category == "age":
        if os.path.exists(METADATA_FILE):
            build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s')
        else:
            build_time = pd.Timestamp.now()
            
        valid_dates = metadata[metadata['parsed_date'] <= build_time].copy()
        oldest = valid_dates.sort_values('parsed_date', ascending=True).head(50)
        newest = valid_dates.sort_values('parsed_date', ascending=False).head(50)
        
        return {
            "top": newest[['appid', 'name', 'release_date']].to_dict(orient='records'),
            "bottom": oldest[['appid', 'name', 'release_date']].to_dict(orient='records')
        }

    elif category == "difficulty":
        hardest = metadata.sort_values('difficulty_predicted', ascending=False).head(50)
        easiest = metadata.sort_values('difficulty_predicted', ascending=True).head(50)
        
        # Tag predictors
        tag_impacts = []
        pred_file = "data/difficulty_predictions.csv"
        if os.path.exists(pred_file):
            pred_df = pd.read_csv(pred_file)
            contrib_cols = [c for c in pred_df.columns if c.startswith('contrib_')]
            if contrib_cols:
                for col in contrib_cols:
                    tag_name = col.replace('contrib_', '').replace('_', ' ').title()
                    impact = pred_df[pred_df[col] != 0][col].mean()
                    if not np.isnan(impact):
                        tag_impacts.append({'tag': tag_name, 'impact': float(impact)})
        
        return {
            "top": hardest[['appid', 'name', 'difficulty_predicted']].to_dict(orient='records'),
            "bottom": easiest[['appid', 'name', 'difficulty_predicted']].to_dict(orient='records'),
            "tag_impacts": sorted(tag_impacts, key=lambda x: x['impact'], reverse=True)
        }

    raise HTTPException(status_code=400, detail="Invalid category")

@app.post("/metadata")
def get_metadata(request: MetadataRequest):
    """Returns metadata for a specific list of game names."""
    if data_manager.metadata is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    metadata = data_manager.metadata
    matches = metadata[metadata['name'].isin(request.names)]
    
    response_items = []
    for _, game_meta in matches.iterrows():
        raw_pop = game_meta['positive'] + game_meta['negative']
        raw_length = game_meta['estimated_playtime'] / 60.0
        
        item = {
            "appid": int(game_meta['appid']),
            "name": str(game_meta['name']),
            "release_date": str(game_meta['release_date']),
            "release_year": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "estimated_playtime": float(game_meta['estimated_playtime']) if pd.notna(game_meta['estimated_playtime']) else 0.0,
            "difficulty_predicted": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0,
            "positive": int(game_meta['positive']),
            "negative": int(game_meta['negative']),
            "genres": game_meta['genres'],
            "tags": game_meta['tags'],
            "raw_pop": int(raw_pop),
            "raw_length": float(raw_length)
        }
        response_items.append(item)
    return response_items

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    if data_manager.metadata is None:
         raise HTTPException(status_code=503, detail="Data not loaded yet")

    metadata = data_manager.metadata
    
    # Identify seeds
    seed_indices = np.where(metadata['name'].isin(request.seed_games))[0]
    seed_appids = metadata.iloc[seed_indices]['appid'].tolist()

    # 1. Filtering
    mask = np.ones(len(metadata), dtype=bool)
    
    if request.remove_vr:
        mask &= ~metadata['is_vr_only'].values
        
    if request.english_only:
        mask &= metadata['is_english'].values
    
    if request.remove_nsfw:
        mask &= (metadata['mature_content'] == 0).values
        
        tags = metadata['tags'].fillna('')
        nsfw_tag_mask = tags.apply(lambda x: any(tag in str(x).lower() for tag in NSFW_TAGS) if x else False).values
        mask &= ~nsfw_tag_mask
        
        nsfw_name_mask = metadata['name'].fillna('').str.contains('|'.join(NSFW_NAME_PATTERNS), case=False, na=False).values
        mask &= ~nsfw_name_mask
    
    if request.remove_utilities:
        mask &= ~metadata['is_utility'].values

    if request.remove_unreleased:
        # Dynamic build time check might be overkill, let's assume current time or file time
        # Using file time for consistency with original logic
        if os.path.exists(METADATA_FILE):
            build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s')
            future_mask = (metadata['parsed_date'] > build_time).fillna(False)
            mask &= ~future_mask.values

    if request.genres:
        # Optimized genre filtering without list column
        # Genres are stored as "['Action', 'RPG']" or "Action, RPG"
        # We check if the string contains the genre.
        # Note: This might match substrings (e.g. "Action" inside "Reaction"), but genre names are usually distinct.
        # To be safe, we could regex, but str.contains is faster.
        # Given the dataset, genre names are standard Steam genres.
        
        # Create a combined mask
        genre_mask = np.zeros(len(metadata), dtype=bool)
        for genre in request.genres:
            # Escape regex chars if any
            escaped_genre = re.escape(genre)
            # Check if genre exists in the string representation
            # Simple contains is usually sufficient for standard genres
            genre_mask |= metadata['genres'].fillna('').astype(str).str.contains(escaped_genre, regex=True, case=False).values
        
        mask &= genre_mask

    keep_indices = np.where(mask)[0]
    
    # 2. Semantic Component
    all_semantic_sims = np.zeros(len(metadata))
    
    if request.prompt or seed_appids:
        if request.prompt:
            prompt_vec = data_manager.model.encode([request.prompt])[0]
            
            p_desc_centered = (prompt_vec - data_manager.mean_desc) if data_manager.mean_desc is not None else prompt_vec
            p_struct_centered = (prompt_vec - data_manager.mean_structural) if data_manager.mean_structural is not None else prompt_vec
            
            p_desc = np.dot(p_desc_centered, data_manager.w_desc) if data_manager.w_desc is not None else p_desc_centered
            p_struct = np.dot(p_struct_centered, data_manager.w_structural) if data_manager.w_structural is not None else p_struct_centered
            
            def norm_vec(v):
                mag = np.linalg.norm(v)
                return v / (mag if mag > EPSILON else 1.0)
            
            p_desc_norm = norm_vec(p_desc)
            p_struct_norm = norm_vec(p_struct)
            
            prompt_desc_sims = np.dot(data_manager.embeddings_desc_norm, p_desc_norm)
            prompt_structural_sims = np.dot(data_manager.embeddings_structural_norm, p_struct_norm)
            
            all_semantic_sims = (prompt_desc_sims + prompt_structural_sims) * SEMANTIC_PROMPT_SEED_BLEND

        if seed_indices.size > 0:
            seed_desc_vecs = data_manager.embeddings_desc_norm[seed_indices]
            avg_seed_desc = np.mean(seed_desc_vecs, axis=0)
            sd_mag = np.linalg.norm(avg_seed_desc)
            sd_norm = avg_seed_desc / (sd_mag if sd_mag > EPSILON else 1.0)
            seed_desc_sims = np.dot(data_manager.embeddings_desc_norm, sd_norm)
            
            seed_structural_vecs = data_manager.embeddings_structural_norm[seed_indices]
            avg_seed_structural = np.mean(seed_structural_vecs, axis=0)
            ss_mag = np.linalg.norm(avg_seed_structural)
            ss_norm = avg_seed_structural / (ss_mag if ss_mag > EPSILON else 1.0)
            seed_structural_sims = np.dot(data_manager.embeddings_structural_norm, ss_norm)
            
            seed_combined_sims = (seed_desc_sims + seed_structural_sims) * SEMANTIC_PROMPT_SEED_BLEND
            
            if request.prompt:
                all_semantic_sims = (all_semantic_sims + seed_combined_sims) * SEMANTIC_PROMPT_SEED_BLEND
            else:
                all_semantic_sims = seed_combined_sims

    all_tag_sims = np.zeros(len(metadata))
    if seed_indices.size > 0:
        tag_seed_vectors = data_manager.tag_vectors[seed_indices]
        combined_tag_query = np.mean(tag_seed_vectors, axis=0)
        tag_q_mag = np.linalg.norm(combined_tag_query)
        
        dot_products = np.dot(data_manager.tag_vectors, combined_tag_query)
        denom = (data_manager.tag_vectors_norms * tag_q_mag) + DOT_PRODUCT_LAMBDA
        denom[denom == 0] = EPSILON
        
        all_tag_sims = dot_products / denom

    semantic_sims = all_semantic_sims[keep_indices]
    tag_sims = all_tag_sims[keep_indices]

    # 4. Rating Component
    num_grid_rows = data_manager.quality_grid.shape[0]
    grid_index = int(round(((request.disc_pref - (-1.0)) / 2.0) * (num_grid_rows - 1)))
    grid_index = max(0, min(num_grid_rows - 1, grid_index))
    
    z_spps = np.clip(data_manager.quality_grid[grid_index][keep_indices], Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    
    z_date = np.clip(metadata.iloc[keep_indices]['date_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_pop = np.clip(metadata.iloc[keep_indices]['pop_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_length = np.clip(metadata.iloc[keep_indices]['playtime_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)
    z_difficulty = np.clip(metadata.iloc[keep_indices]['difficulty_z'].values, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX)

    # 5. Hybrid Scoring
    gamma = QUALITY_WEIGHT_MULTIPLIER * request.quality_pref
    
    w_semantic = request.alpha if (request.prompt or seed_appids) else 0.0
    w_tag = request.beta if seed_appids else 0.0
    w_spps = gamma
    w_date = AGE_WEIGHT_MULTIPLIER * request.age_pref
    w_pop = POPULARITY_WEIGHT_MULTIPLIER * request.pop_pref
    w_length = LENGTH_WEIGHT_MULTIPLIER * request.length_pref
    w_difficulty = DIFFICULTY_WEIGHT_MULTIPLIER * request.difficulty_pref

    z_semantic = np.clip(to_z(semantic_sims), Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) if (request.prompt or seed_appids) else np.zeros(len(keep_indices))
    z_tag = np.clip(to_z(tag_sims, ignore_zeros=True), Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX) if seed_appids else np.zeros(len(keep_indices))

    final_scores = calculate_hybrid_score(
        z_semantic, w_semantic,
        z_tag, w_tag,
        z_spps, w_spps,
        z_date, w_date,
        z_pop, w_pop,
        z_length, w_length,
        z_difficulty, w_difficulty
    )

    # Exclude seeds
    meta_filt = metadata.iloc[keep_indices].copy()
    if seed_appids:
        seed_mask = meta_filt['appid'].isin(seed_appids)
        final_scores[seed_mask] = -1e12

    # 6. Sorting and Result Formatting
    num_to_extract = min(len(final_scores), request.top_k * TOP_K_SORT_MULTIPLIER)
    
    if num_to_extract > 0:
        partitioned_indices = np.argpartition(-final_scores, num_to_extract-1)[:num_to_extract]
        subset_scores = final_scores[partitioned_indices]
        subset_names = meta_filt['name'].fillna("").values[partitioned_indices]
        subset_sorted_indices = np.lexsort((subset_names, -subset_scores))
        top_indices = partitioned_indices[subset_sorted_indices[:request.top_k]]
    else:
        top_indices = []

    results = meta_filt.iloc[top_indices].copy()
    
    response_items = []
    
    for i, idx in enumerate(top_indices):
        game_meta = results.iloc[i]
        
        raw_pop = game_meta['positive'] + game_meta['negative']
        raw_length = game_meta['estimated_playtime'] / 60.0
        
        item = {
            "appid": int(game_meta['appid']),
            "name": str(game_meta['name']),
            "release_date": str(game_meta['release_date']),
            "release_year": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "estimated_playtime": float(game_meta['estimated_playtime']) if pd.notna(game_meta['estimated_playtime']) else 0.0,
            "difficulty_predicted": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0,
            "positive": int(game_meta['positive']),
            "negative": int(game_meta['negative']),
            "genres": game_meta['genres'], 
            "tags": game_meta['tags'],     
            
            "weighted_score": float(final_scores[idx]),
            "semantic_match": float(semantic_sims[idx]),
            "tag_match": float(tag_sims[idx]),
            "rating": float(z_spps[idx]), 
            
            "z_semantic": float(z_semantic[idx]),
            "w_semantic": float(w_semantic),
            "z_tag": float(z_tag[idx]),
            "w_tag": float(w_tag),
            "z_spps": float(z_spps[idx]),
            "w_spps": float(w_spps),
            "z_date": float(z_date[idx]),
            "w_date": float(w_date),
            "z_pop": float(z_pop[idx]),
            "w_pop": float(w_pop),
            "z_length": float(z_length[idx]),
            "w_length": float(w_length),
            "z_difficulty": float(z_difficulty[idx]),
            "w_difficulty": float(w_difficulty),
            
            "raw_date": int(game_meta['release_year']) if pd.notna(game_meta['release_year']) else 0,
            "raw_pop": int(raw_pop),
            "raw_length": float(raw_length),
            "raw_difficulty": float(game_meta['difficulty_predicted']) if pd.notna(game_meta['difficulty_predicted']) else 0.0
        }
        response_items.append(item)
        
    return response_items

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
