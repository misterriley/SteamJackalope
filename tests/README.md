# Tests Directory

This directory contains automated tests for the SteamJackalope application, ensuring code quality, correctness, and preventing regressions.

## Overview

The test suite covers:

- API endpoints (backend server)
- Data processing pipelines
- Utility functions
- Memory constraints
- Scraping robustness
- Tag and embedding generation

## Running Tests

### Convenience Script

```bash
run_all_tests.bat
```

### Manual Execution

```bash
pytest tests/
```

For verbose output:

```bash
pytest tests/ -v
```

For specific test files:

```bash
pytest tests/test_utils.py
```

## Test Isolation

**Critical:** All tests must use temporary files or mock objects to avoid modifying production data artifacts. The `common/constants.py` module supports environment variable overrides for all data file paths. Tests should set these to point to temporary directories created during the test.

Example pattern:

```python
import os
import tempfile
import pytest

def test_something():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['STEAM_METADATA_FILE'] = os.path.join(tmpdir, 'test_metadata.parquet')
        # Run test without touching production files
```

## Test Categories

### Unit Tests

- `test_utils.py` - Utility functions (z-scoring, hybrid scoring)
- `test_regularization.py` - Regularization constant calculations
- `test_tag_transformations.py` - Tag vector transformations
- `test_date_z_scores.py` - Date z-score calculations

### Integration Tests

- `test_metadata_processing.py` - Metadata generation pipeline
- `test_metadata_sync.py` - Synchronization between scraped data and pipeline
- `test_pipeline_integration.py` - End-to-end pipeline stages
- `test_tag_vector_generation.py` - Full tag vector pipeline

### API Tests

- `test_metadata_api.py` - `/metadata` endpoint
- `test_genre_filter_api.py` - `/genres` endpoint and filtering
- `test_lists_api.py` - `/lists` endpoint
- `test_app_launch.py` - Server startup and basic functionality

### Robustness Tests

- `test_tag_vectors_robustness.py` - Edge cases in tag vector handling
- `test_scraping_robustness.py` - Error handling in scraping operations
- `test_memory_constraints.py.bak` - Memory usage validation (backup)

### Model Tests

- `test_difficulty_pipeline.py` - Difficulty model training and predictions
- `test_em_pipeline.py` - EM imputation algorithm

## Test Fixtures

Common test fixtures are defined in individual test files. There is no central `conftest.py` at present, but one could be added if fixture sharing becomes necessary.

## Continuous Integration

The project aims for high test coverage. When adding new features:

1. Write tests that cover the new functionality
2. Ensure all existing tests pass
3. Follow the isolation pattern to protect production data

## Troubleshooting

### Pytest Collection Errors

If tests fail to collect, check for naming collisions with files in `old/` directory. Legacy scripts with module names matching test files can cause conflicts. Rename or move problematic files.

### Memory Issues During Tests

Some tests create temporary data files. Ensure these are properly cleaned up. Use `tmp_path` fixture from pytest for file operations.

## Related

- See `pipeline/README.md` for production pipeline components that these tests validate
- See `app/README.md` for backend endpoints being tested
- See `common/README.md` for utility functions under test