# Changelog

## [2026-03-13] - Visual Context & Interactive Media (v7.2)
### Build 56
### Added
- **High-Signal Hover Preview**:
    - Implemented a rich, interactive hover card across all recommendation and search views.
    - **Automated Media Slideshow**: Real-time trailers and screenshots fetched from Steam with intelligent auto-advancement (2.5s for images, auto-next for trailers).
    - **Silent Default Playback**: Trailers play automatically and silently with a manual sound toggle to ensure a professional, non-intrusive browsing experience.
    - **Magnitude-Sorted Breakdown**: A dynamic, color-coded chart visualizing exactly how each mathematical feature (Quality, Age, Kernel, etc.) contributed to the game's final score.
    - **Raw Stat Grid**: Instant access to ground-truth **Price**, **Difficulty**, and **Estimated Length** (in hours).
- **Stable Hover Pattern**: Unified hover logic across Recommender, DNA Solver, and Catalogue views with a 150ms intent delay and React Portals for robust viewport clamping.
- **Backend Media Caching**:
    - Added `/games/{appid}/media` endpoint with disk-based caching in `data/media_cache/`.
    - Bridges Steam's AppDetails API to provide high-speed media links while respecting rate limits.
- **Interactive Data Sync**:
    - Added a **"Sync Data"** button to the Interactive Rankings panel to force-reload the taste profile and bypass browser caching.
    - Integrated live UI weight updates into the hover card's contribution chart for real-time visual feedback.

### Changed
- **Restored DNA Solver Workflow**: Re-implemented the 3-step acquisition process (**Acquire -> Verify -> Solve**) to restore the high-fidelity user taste profile experience.
- **Extended Metadata Pipeline**:
    - Updated `/metadata` and `/recommend` endpoints to serve raw price, difficulty, and playtime data for the new hover card grid.
    - Enhanced `solve_user_taste.py` to inject feature-level z-scores into the `interactive_pool` for high-fidelity score breakdowns in the interactive view.
- **Improved Steam API Resilience**:
    - Refined media fetching logic to handle modern HLS/DASH trailer formats and bypass success:false responses caused by overly restrictive filters.

### Fixed
- **Solver Scalar Z-Scoring**: Corrected a critical math error in `solve_user_taste.py` where scalar z-scoring was zeroing out all features; restored profile accuracy and recommendation relevance.
- **Production Build Recovery**: Resolved 35+ TypeScript and ESLint errors blocking the Vite production build; synchronized the development environment with the production `dist/` folder.

## [2026-03-06] - High-Fidelity Similarity & Puzzle Firewall (Research)
... [rest of changelog] ...
