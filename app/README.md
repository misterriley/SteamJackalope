# App Directory

This directory contains the application code for both the backend API server and the Streamlit frontend.

## Components

### Backend (`server.py`)

FastAPI server that provides:

- `/recommend` - Main recommendation endpoint with hybrid scoring
- `/metadata` - Retrieve full metadata for specific games
- `/genres` - Get unique list of available genres
- `/lists` - Cached endpoints for quality, popularity, and similarity lists

The backend handles all heavy data loading and vector computations:

- Loads embeddings, tag vectors, and metadata with memory mapping
- Performs z-score calculations and hybrid score blending
- Supports genre filtering with OR logic
- Implements server-side caching for performance

### Frontend (`app.py`)

Streamlit UI that communicates with the backend:

- Natural language prompt input
- Seed game selection with search
- Slider controls for weighting factors (quality, discovery, popularity, age, length, difficulty)
- Multi-genre filtering
- Debug mode showing detailed score breakdowns
- Lists page for extreme rankings and similarity analysis

### Launcher (`lists.py`)

Helper module for the lists page that generates curated rankings:

- Highest/lowest quality games
- Longest/shortest experiences
- Predicted difficulty extremes
- Similarity analysis for diverse popular seeds

## Running Locally

Start the backend:

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

In a separate terminal, start the frontend:

```bash
streamlit run app/app.py
```

Or use the convenience script:

```bash
run_test_env.bat
```

## Architecture Notes

- The backend uses lazy loading: heavy dependencies (SentenceTransformer, torch) only load when text prompts are used
- All large NumPy arrays are memory-mapped (`mmap_mode='r'`) to keep RAM usage minimal
- Metadata uses PyArrow backend for efficient string storage
- Pre-normalized embeddings enable fast cosine similarity via dot product
- Caching is used extensively (`st.cache_data` in frontend, custom cache in backend) to avoid redundant operations

## Configuration

Data file paths are configurable via environment variables (see `common/constants.py` for all available overrides). Tests use temporary paths to avoid modifying production artifacts.

## Troubleshooting

- If connection errors occur on first load, check that the backend is fully started before refreshing the frontend
- For memory issues in production, verify that all `.npy` files are being memory-mapped and not loaded fully into RAM
- Ensure `metadata.parquet` is in the root directory, not in subfolders

## Related

- See `orientation.md` for overall project context
- See `methodology.md` for detailed explanations of the scoring algorithms
- See `common/utils.py` for shared utility functions like z-scoring and hybrid calculation
