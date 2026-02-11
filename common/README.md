# Common Directory

This directory contains shared utilities, constants, and helper functions used across the entire SteamJackalope project.

## Components

### `constants.py`

Centralized configuration and hyperparameters for the entire application:

- Data file paths (with environment variable overrides for testing)
- Regularization constants (loaded from `pipeline/regularization_constants.json`)
- Model parameters (z-score clamps, whitening settings, etc.)
- Global constants used in multiple modules

**Important:** All file paths support environment variable overrides (e.g., `STEAM_METADATA_FILE`) to enable test isolation. Never hardcode paths in other modules; always import from `common.constants`.

### `utils.py`

Shared utility functions:

- `to_z` - Convert probabilities to z-scores with numerical stability (uses float64)
- `zscore_clamp` - Clamp z-scores to prevent outlier domination
- `hybrid_score` - Calculate weighted combination of normalized components
- Various data transformation helpers

## Usage

```python
from common.constants import (
    METADATA_FILE,
    TAG_VECTOR_K,
    GLOBAL_POSITIVE_RATE,
    Z_SCORE_CLAMP_MIN,
    Z_SCORE_CLAMP_MAX
)
from common.utils import to_z, hybrid_score

# Use in your code
z = to_z(0.75, prior=0.5, pseudo_count=1)  # Bayesian z-score
score = hybrid_score([semantic_z, tag_z, quality_z], weights=[1.0, 0.5, 0.3])
```

## Environment Variables

The following environment variables can override default paths:

- `STEAM_METADATA_FILE` - Override `metadata.parquet` location
- `STEAM_EMBEDDINGS_DESC_FILE` - Override description embeddings
- `STEAM_EMBEDDINGS_STRUCTURAL_FILE` - Override structural embeddings
- `STEAM_TAG_VECTORS_FILE` - Override tag vectors
- `STEAM_TAG_VECTORS_NORMS_FILE` - Override tag vector norms
- `STEAM_QUALITY_SCORES_FILE` - Override quality grid
- `RAW_DOWNLOAD_PATH` - Override raw scraped data cache location (outside repo by default)

See `constants.py` for complete list.

## Testing Isolation

When writing tests, always use environment variables to point to temporary files to avoid modifying production data. The test suite uses this pattern extensively.

## Notes

- All regularization constants (e.g., `TAG_VECTOR_K`, `DOT_PRODUCT_LAMBDA`) are automatically calculated during pipeline execution and read from `pipeline/regularization_constants.json` at startup.
- The `to_z` function automatically uses `dtype=np.float64` to prevent numerical overflow on large float16 arrays.
- Constants defined here are imported by both the app server and pipeline scripts, ensuring consistency across the stack.

## Related

- See `orientation.md` for project overview
- See `methodology.md` for explanations of the statistical models that use these constants