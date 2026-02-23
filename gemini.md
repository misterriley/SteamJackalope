# Gemini Project Guide: SteamJackalope

This document serves as a condensed reference for Gemini instances working on this repository. It synthesizes critical operational knowledge from `onStartup.md`, `orientation.md`, and `onShutdown.md`.

## 🚀 Quick Start Workflow
1.  **Orient**: Read `orientation.md` (architecture), `methodology.md` (math/stats), and `user.md` (user expectations and working style).
2.  **Sync**: Ensure you are working with the latest artifacts. Production data (`metadata.parquet`, `.npy` files) should be in `data/production/` and synchronized.
3.  **Tasking**: Check `tasklist.md` for active tasks. If empty, check `ideas.md` to propose new ones.
4.  **Environment**: 
    *   Backend: `python -m uvicorn app.server:app --host 127.0.0.1 --port 8000`
    *   Frontend (Modern): `cd frontend; npm run dev`
    *   Frontend (Legacy): `streamlit run app/app.py`
    *   Convenience: Use `run_test_env.bat` to launch both backend and modern frontend.

## 🏗️ Architecture & Tech Stack
*   **Decoupled Design**: FastAPI Backend (`app/server.py`) + Modern React 19 Frontend (`frontend/`) or Legacy Streamlit Frontend (`app/app.py`).
*   **Data Storage**: Production artifacts (`metadata.parquet`, `.npy` files) are stored in `data/production/`.
*   **Memory Optimization**: Use `mmap_mode='r'` for all large NumPy arrays.
    *   Use `float16` for storage but `float64` for statistical calculations (`mean`, `std`) to avoid overflow.
*   **Vectorization**: Use `numpy` for all scoring and similarity logic. Avoid Python loops in the hot path.

## ⚠️ Critical Constraints & Gotchas
*   **Artifact Synchronization**: All `.npy` files and `metadata.parquet` must have matching row counts and ordering, derived from `data/pipeline_games_clean.csv`.
*   **Path Management**: `common/constants.py` is the single source of truth for paths and magic numbers. Use environment variables (e.g., `STEAM_METADATA_FILE`) to override for tests.
*   **Parquet Schema**: Use `pyarrow.parquet.read_schema(file).names` to check columns without loading data.
*   **Hybrid Semantic Model**: Uses a 235-dimensional ZCA-whitened descriptive space. All text must be **lowercased** before encoding to maintain parity with production artifacts.
*   **Variance Parity**: Semantic embeddings are scaled by **11.25x** to match tag variance in the LASSO solver.
*   **Tag Vectors**: Use PCA-ZCA whitening to prevent similarity explosion. `TAG_VECTOR_K` requires LOD imputation in its solver loop.
*   **Scraping**: Requires `STEAM_API_KEY`. Uses a hierarchical local cache outside the repo.

## 🧪 Testing & Quality
*   **Runner**: Use `run_all_tests.bat` (wraps `pytest`).
*   **Production Lock**: The test runner automatically locks the `data/production` directory as **Read-Only** during execution. This prevents tests from accidentally corrupting production artifacts (e.g., `tag_names.json`). Any test attempting to write to this directory will fail with a `PermissionError`.
*   **Isolation**: Never modify production artifacts during tests. Use temporary files or environment overrides.
*   **Logging**: Ensure all new functionality includes informative logging.

## 🏁 Shutdown Protocol (Before finishing a session)
1.  **Update Docs**: If you changed logic, update `methodology.md`, `orientation.md`, and potentially `onPush.md`.
2.  **Changelog**: Review `onPush.md` and update `CHANGELOG.md` with your changes (Algorithm, UI, or Deployment).
3.  **Clean Up**: Remove temporary files. Ensure no `>>>>+++ REPLACE` markers remain.
4.  **Externalize**: Move any new magic numbers/paths to `constants.py`.
5.  **Verify**: Run the full test suite.
6.  **Commit**: Group changes logically with a "why-focused" message. Do not push.
7.  **Tasklist**: Move completed tasks to "Recently Completed" in `tasklist.md`.

## 📂 Directory Map
*   `app/`: UI and Server logic.
*   `pipeline/`: Data processing and artifact generation.
*   `common/`: Constants and shared utilities.
*   `scraping/`: Steam data collection.
*   `research/`: Statistical analysis and experimental scripts.
*   `tests/`: Automated test suite.
*   `data/`: Intermediate data (Source of truth: `pipeline_games_clean.csv`).
*   `root/`: Contains `onPush.md` (Push Protocol) and `CHANGELOG.md`.
