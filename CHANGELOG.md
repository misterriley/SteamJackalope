# Changelog

All notable changes to the Steam Jackalope project will be documented in this file.

## 2026-02-14
### Build 11
### Version 0.0.1-pre-alpha+build.11
- **Documentation**: Integrated `onPush.md` into `orientation.md`, `onShutdown.md`, and `gemini.md` for better onboarding.
- **UI**: Added a "Changelog" link to the header in both modern and legacy frontends.
- **Documentation**: Established a new versioning and changelog protocol, including `onPush.md` and automated build increments.
- **Tooling**: Refactored `generate_random_recommendation.py` output to a multiline, labeled format optimized for Discord sharing.
- **Algorithm**: Updated difficulty display to a 10-point scale for more granular feedback in CLI recommendations.

## 2026-02-13
### Build 3-8
### Version 0.0.1-pre-alpha+build.3
- **Frontend**: Launched a modern React 19 + TypeScript + Vite + Tailwind CSS v4 frontend, replacing the legacy Streamlit interface as the primary UI.
- **Architecture**: Implemented a Unified Server architecture where FastAPI serves both the API and the production frontend build from a single process.
- **UI**: Added "About" and "Methodology" pages with KaTeX support for mathematical formulas.
- **UI**: Improved seed game management with removal buttons and better grid alignment.
- **Deployment**: Added automated Linux deployment scripts (`deploy.sh`) for unified server management.

## 2026-02-12
- **Algorithm**: Optimized playtime sentiment analysis using full vectorization and parallelized global parameter optimization ($\gamma \approx 0.51$).
- **UI**: Fixed a bug where difficulty scores of 0 were not being displayed in game cards.
- **UI**: Added color-coded weighting visualizations to game cards (Lesser Priority task).

## 2026-02-11
- **UI**: Split the methodology page into dedicated "About" and "Methodology" sections for better readability.
- **UI**: Implemented a compact sidebar layout and 3-column genre selection to maximize vertical space.
- **UI**: Added a custom header and footer with GitHub integration and copyright information.
- **Documentation**: Added detailed README files for all major project directories.

## 2026-02-10
- **Algorithm**: Implemented alphabetical fallback sorting for recommendations when all preference sliders are set to zero.
- **UI**: Refactored all preference sliders (Age, Quality, Popularity, etc.) to use discrete steps for more precise user control.
- **UI**: Added "Debug Mode" to game cards, providing a full breakdown of z-scores and weight contributions for transparency.

## 2026-02-09
- **Performance**: Migrated to quantized ONNX transformer models (`all-MiniLM-L6-v2`), reducing RAM usage and increasing inference speed.
- **Performance**: Converted large production artifacts to float16 precision, reducing the memory footprint by 50%.
- **Algorithm**: Applied dimensionality reduction to whitening matrices to optimize cache performance and reduce noise in high-dimensional similarities.

## 2026-02-14 (Build 12)
- **Algorithm**: Re-calibrated playtime sentiment hyperparameters ($\gamma \approx 0.688$, $s \approx 1.453$) using a full-dataset parallelized grid search (36k+ games) with per-review cross-entropy loss.
- **Infrastructure**: Removed the 512 MB RAM constraint across all documentation, tests, and deployment scripts.
- **Backend**: Switched `SentenceTransformer` from lazy to eager loading at startup to improve initial request latency.
- **Frontend**: Added SEO metadata (Open Graph, Twitter Cards) to the landing page.
- **UI**: Integrated Steam Store links into the "Lists" view for direct navigation.
- **UI**: Added "Trending" seed game selection option in the Recommendations view.
- **Scraping**: Implemented `scrape_trending.py` to fetch top 100 Steam games daily.

## 2026-02-08
- **Deployment**: Achieved a highly efficient RAM footprint for the backend server, enabling deployment on standard cloud tiers.
- **Algorithm**: Implemented thresholding in the z-score normalization function to mitigate noise from dense tag vectors.
- **Algorithm**: Resolved a critical tag similarity bug and eliminated genre contamination in semantic vector matching.
- **Infrastructure**: Configured Git LFS for production data files to ensure repository stability and efficient deployment.

## 2026-02-07
- **Algorithm**: Initial implementation of the hybrid recommendation engine combining semantic search, PCA-ZCA whitened tag vectors, and Bayesian quality scoring.
- **Algorithm**: Added support for multi-genre filtering in the recommendation loop.
- **UI**: Initial release of the legacy Streamlit frontend with interactive sliders and real-time result updates.
- **Scraping**: Established robust Steam storefront scraping architecture with hierarchical caching and automatic checkpointing.
