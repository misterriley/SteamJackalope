# Pipeline Directory

This directory contains scripts for generating and updating the data artifacts used by the recommendation engine.

## Overview

The pipeline transforms raw scraped data into processed embeddings, tag vectors, and quality scores. It uses a modular design where each stage generates a specific artifact.

## Main Script

### `run_pipeline.py`

Orchestrates the entire pipeline execution:

1. Generate tag vectors (`generate_tag_vectors.py`)
2. Generate semantic embeddings (`generate_semantic_vectors.py`)
3. Generate metadata (`generate_metadata.py`)
4. Generate quality scores grid (`generate_quality_scores_grid.py`)
5. Calculate regularization constants (`calculate_regularization.py`)

Configuration is read from `pipeline_config.json` for file paths and processing intervals.

## Pipeline Stages

### `generate_tag_vectors.py`

Transforms raw Steam tags into regularized, whitened embeddings:

- Iterative EM imputation for censored tags (top 20 limit)
- Bayesian smoothing with optimized `TAG_VECTOR_K`
- Transformation (Anscombe/CLR/Identity) with centering
- Truncated PCA-ZCA whitening to 128 dimensions
- Outputs: `steam_tag_vectors.npy`, `tag_vectors_norms.npy`

### `generate_semantic_vectors.py`

Creates dual semantic embeddings from game text:

- Structural vectors (genres + tags)
- Descriptive vectors (description + reviews)
- Applied with all-MiniLM-L6-v2 model
- ZCA whitening with saved transformation matrices
- Outputs: `embeddings_structural.npy`, `embeddings_desc.npy`, `mean_*.npy`, `w_*.npy`

### `generate_metadata.py`

Builds the central metadata parquet file:

- Merges scraped data with pipeline outputs
- Calculates z-scores for date, popularity, playtime
- Repairs stale review counts from individual reviews
- Parses genres/tags into lists
- Outputs: `metadata.parquet`

### `generate_quality_scores_grid.py`

Produces Bayesian quality scores across discovery settings:

- Grid of 201 steps from low to high discovery
- Pre-normalized z-score transformations
- Optional distribution pinning for consistency
- Outputs: `quality_scores_grid.npy`

### `calculate_regularization.py`

Derives data-driven constants:

- `TAG_VECTOR_K` - Tag smoothing parameter via stochastic path analysis
- `GLOBAL_POSITIVE_RATE` - Overall Steam review positivity
- `DOT_PRODUCT_LAMBDA` - Regularization for tag similarity from Chi distribution
- `PLAYTIME_C` - Playtime regularization constant
- Outputs: `regularization_constants.json`

## Configuration

Edit `pipeline/pipeline_config.json` to customize:

- Input/output file paths
- Processing intervals (how many games to process at once)
- Memory mapping options
- Transformation type for tag vectors
- Quality grid resolution

## Running the Pipeline

```bash
python pipeline/run_pipeline.py
```

For manual execution of individual stages:

```bash
python pipeline/generate_tag_vectors.py data/pipeline_games_clean.csv
python pipeline/generate_semantic_vectors.py
python pipeline/generate_metadata.py data/pipeline_games_clean.csv
python pipeline/generate_quality_scores_grid.py
python pipeline/calculate_regularization.py
```

## Important Notes

- The pipeline uses `data/pipeline_games_clean.csv` as the source of truth for appids. All outputs are synchronized to this list.
- Games with missing names are filtered out to ensure consistency.
- All regularization constants are recalculated during each full pipeline run.
- Memory mapping is used wherever possible for efficiency.
- The `common/constants.py` module loads `pipeline/regularization_constants.json` at startup to make constants available to the app.

## Output Files

Generated artifacts are stored in the project root by default (for Git LFS tracking):

- `steam_tag_vectors.npy` - Regularized tag embeddings (128D)
- `tag_vectors_norms.npy` - Pre-calculated L2 norms
- `embeddings_structural.npy`, `embeddings_desc.npy` - Dual semantic vectors
- `mean_*.npy`, `w_*.npy` - Whitening parameters
- `metadata.parquet` - Game metadata with z-scores
- `quality_scores_grid.npy` - Precomputed quality scores
- `regularization_constants.json` - Auto-calculated hyperparameters

## Related

- See `methodology.md` for theoretical background on each stage
- See `orientation.md` for project overview and current status
- See `common/constants.py` for how these artifacts are consumed by the app