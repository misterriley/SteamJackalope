# Tools Directory

This directory contains utility scripts for debugging, maintenance, and validation of the SteamJackalope data artifacts and analysis workflows.

## Overview

Tools are standalone utilities that support:

- Vector analysis and interrogation
- Pre-computation of derived values
- Dimensionality verification
- Database matching and lookups

These scripts are typically run ad-hoc during development or troubleshooting, not as part of the regular pipeline.

## Components

### `interrogate_vectors.py`

Interactive Streamlit GUI for inspecting vector distributions and similarities:

- View histogram of tag vector lengths or cosine similarities
- Compare specific games to see their similarity rankings
- Test prompt matching against semantic embeddings
- Diagnose vector quality and spacing

Run with:

```bash
streamlit run tools/interrogate_vectors.py
```

### `precalculate_similarity.py`

Precompute similarity matrices for specific seed games or categories. Used for generating similarity lists or caching expensive pairwise calculations.

### `precalculate_tag_norms.py`

Regenerates `tag_vectors_norms.npy` from `steam_tag_vectors.npy`. Use if the norms file becomes out of sync with the tag vectors (the server can detect and handle this, but precalculating avoids runtime overhead).

### `precalculate_norm_embeddings.py`

Normalizes semantic embedding matrices to unit length. Typically run once after generating embeddings, or if re-normalization is needed.

### `check_dimension_reduction.py`

Validates that dimensionality reduction (PCA/ZCA whitening) was applied correctly. Checks preserved variance, singular values, and output dimensions.

### `check_grid.py`

Verifies the quality scores grid structure and consistency. Can check for alignment between different discovery settings.

### `match_game_to_database.py`

Helper for looking up game metadata by name or appid. Useful for debugging mismatches between datasets.

## When to Use These Tools

- **Debugging vector issues**: Use `interrogate_vectors.py` to examine distributions and spot anomalies
- **Performance optimization**: Precalculate norms or similarities to speed up repeated operations
- **Validation**: Run check scripts after pipeline updates to ensure artifact integrity
- **Data exploration**: Quick lookups and comparisons during development

## Important Notes

- These tools are **not** part of the production pipeline; they are for developer use
- Some tools may load large arrays into memory; be mindful of RAM usage
- Outputs from these tools typically go to console or interactive UI, not to production files
- The server has built-in robustness checks (e.g., auto-recalculating norms if mismatched), so these tools are optional but helpful

## Related

- See `pipeline/README.md` for the main data generation pipeline
- See `app/README.md` for how these artifacts are consumed by the recommendation engine
- See `research/README.md` for more experimental analysis scripts