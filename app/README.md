# App Directory

This directory contains the **Backend API Server** and a **Legacy Streamlit Frontend**.

## Components

### Backend (`server.py`)

FastAPI server that provides:

- `/recommend` - Main recommendation endpoint with hybrid scoring
- `/metadata` - Retrieve full metadata for specific games
- `/genres` - Get unique list of available genres
- `/lists` - Cached endpoints for quality, popularity, and similarity lists
- `/games/search` - Fast autocomplete for game selection
- `/games/random` - Returns a random game name

The backend handles all heavy data loading and vector computations:

- Loads embeddings, tag vectors, and metadata with memory mapping
- Performs z-score calculations and hybrid score blending
- Supports genre filtering with OR logic
- Implements server-side caching for performance

### Legacy Frontend (`app.py`)

Streamlit UI that communicates with the backend. 
**Note:** For the primary user experience, use the modern frontend in the `frontend/` directory.

### Helper Module (`lists.py`)

Helper module for generating curated rankings:

- Highest/lowest quality games
- Longest/shortest experiences
- Predicted difficulty extremes
- Similarity analysis for diverse popular seeds

## Running Locally

Start the backend:

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

In a separate terminal, start the legacy frontend:

```bash
streamlit run app/app.py
```

## Architecture Notes

- The backend uses lazy loading: heavy dependencies (SentenceTransformer, torch) only load when text prompts are used.
- All large NumPy arrays are memory-mapped (`mmap_mode='r'`) to keep RAM usage minimal.
- Metadata uses PyArrow backend for efficient string storage.
- Pre-normalized embeddings enable fast cosine similarity via dot product.
- Caching is used extensively to avoid redundant operations.
