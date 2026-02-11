# SteamJackalope

A sophisticated Steam game recommendation engine that combines semantic search, tag analysis, and Bayesian quality scoring to provide highly relevant game suggestions.

## Overview

SteamJackalope is a hybrid recommendation system that uses natural language processing and statistical modeling to match users with games based on multiple factors:

- **Semantic similarity** using Sentence Transformers (dual vectors for structure and description)
- **Tag-based similarity** using regularized, whitened tag embeddings
- **Bayesian quality scoring** based on Steam reviews with tunable discovery
- **Additional factors**: popularity, release date, difficulty, and playtime length

The system uses a decoupled architecture with a FastAPI backend for heavy computations and a Streamlit frontend for the user interface.

## Quick Start

### Local Development

1. Clone the repository and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables (copy `.env.example` to `.env` and add your Steam API key)

3. Start the backend server:

   ```bash
   python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
   ```

4. In a separate terminal, start the frontend:

   ```bash
   streamlit run app/app.py
   ```

5. Open browser to `http://localhost:8501`

### Running Tests

Use the convenience script:

```bash
run_all_tests.bat
```

Or manually:

```bash
pytest tests/
```

### Data Pipeline

To regenerate data artifacts (embeddings, tag vectors, quality scores):

```bash
python pipeline/run_pipeline.py
```

See `pipeline/README.md` for detailed documentation.

## Project Structure

This repository is organized into specialized directories:

- **`app/`** - Backend (FastAPI) and frontend (Streamlit) application code
- **`common/`** - Shared utilities, constants, and helper functions
- **`pipeline/`** - Data processing and artifact generation scripts
- **`scraping/`** - Steam data collection and update scripts
- **`research/`** - Experimental scripts and data analysis notebooks
- **`tests/`** - Automated test suite
- **`tools/`** - Utility scripts for debugging and maintenance
- **`deployment/`** - Deployment configurations and scripts
- **`data/`** - Processed data files (some production artifacts stored separately)

See individual directory README files for detailed documentation.

## Key Features

### Hybrid Scoring System

Combines multiple signals with user-tunable weights:

- Semantic search (natural language prompts)
- Tag similarity (user-generated tags)
- Bayesian quality scores (review-based)
- Discovery slider (tunable trust in small reviews)
- Popularity, age, difficulty, and length filters

### Memory Optimization

- Memory-mapped data files for low RAM usage (<512 MB)
- FP16 precision for large matrix operations
- Lazy loading of heavy models (SentenceTransformer)

### Production Ready

- Docker containerization
- Render deployment configuration
- Comprehensive test coverage
- Robust error handling and logging

## Methodology

For detailed information about the statistical models and algorithms used, see:

- `methodology.md` - Technical deep dive into the recommendation engine
- `orientation.md` - Comprehensive project overview and current status

## Contributing

See `onStartup.md` for guidelines for new contributors. Core development workflow:

1. Read `orientation.md` and `methodology.md`
2. Check `tasklist.md` for tasks
3. Consult `ideas.md` if no specific tasks exist
4. Follow testing isolation practices (use environment variable overrides)
5. Update documentation when adding features

## Notes

- Production data files are tracked with Git LFS
- Raw scraped data cached outside repository by default (see `common/constants.py`)
- All regularization constants auto-calculated during pipeline execution
- `metadata.parquet` should reside in root directory only

## License

This project is open source. See repository for license details.

## Links

- [GitHub Repository](https://github.com/misterriley/SteamJackalope)
- [Orientation Guide](orientation.md)
- [Methodology](methodology.md)