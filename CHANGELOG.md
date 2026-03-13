# Changelog

## [2026-03-13] - Visual Context & Interactive Media (v7.2)
### Build 55
### Added
- **High-Signal Hover Preview**:
    - Implemented a rich, interactive hover card across all recommendation and search views.
    - **Automated Media Slideshow**: Real-time trailers and screenshots fetched from Steam with intelligent auto-advancement (2.5s for images, auto-next for trailers).
    - **Silent Default Playback**: Trailers play automatically and silently with a manual sound toggle to ensure a professional, non-intrusive browsing experience.
    - **Magnitude-Sorted Breakdown**: A dynamic, color-coded chart visualizing exactly how each mathematical feature (Quality, Age, Kernel, etc.) contributed to the game's final score.
    - **Raw Stat Grid**: Instant access to ground-truth **Price**, **Difficulty**, and **Estimated Length** (in hours).
- **Backend Media Caching**:
    - Added `/games/{appid}/media` endpoint with disk-based caching in `data/media_cache/`.
    - Bridges Steam's AppDetails API to provide high-speed media links while respecting rate limits.
- **Interactive Data Sync**:
    - Added a **"Sync Data"** button to the Interactive Rankings panel to force-reload the taste profile and bypass browser caching.
    - Integrated live UI weight updates into the hover card's contribution chart for real-time visual feedback.

### Changed
- **Extended Metadata Pipeline**:
    - Updated `/metadata` and `/recommend` endpoints to serve raw price, difficulty, and playtime data for the new hover card grid.
    - Enhanced `solve_user_taste.py` to inject feature-level z-scores into the `interactive_pool` for high-fidelity score breakdowns in the interactive view.
- **Improved Steam API Resilience**:
    - Refined media fetching logic to handle modern HLS/DASH trailer formats and bypass success:false responses caused by overly restrictive filters.

## [2026-03-06] - High-Fidelity Similarity & Puzzle Firewall (Research)
### Added
- **Optimized Similarity Function**:
    - Created an evolutionary script (`auto_optimizer.py`) to determine the optimal weights for the 4 core similarity metrics: Descriptions (0.445), Verbs (0.233), Tags (0.174), and Graph Links (0.148).
    - Reduced popularity bias by heavily discounting Graph Similarity based on `pop_z` score.
    - Implemented a robust "shovelware" filter requiring at least 63 reviews and >= 65% positive ratio, blocking asset flips and abandoned projects from polluting top results.
- **Puzzle Subgenre Firewall**:
    - Addressed severe semantic bleed in the puzzle genre where 3D spatial games were recommended alongside 2D grid/sokoban games due to pure semantic similarity.
    - Created `test_similarity_firewall.py` in `/research` with a structural penalty loop mapping games to mutually exclusive types: Hidden Object, Sokoban/Grid, Automation, and Spatial/3D.

## [2026-03-05] - Right-Click Categorization & Robust Art
### Added
- **Integrated Context Menu**:
    - Implemented a checkmark-enabled right-click menu across **Solve**, **Catalogue**, and **Recommender** pages.
    - Allows real-time categorization into Backlog, Wishlist, Played, Rated, or Ignored.
    - Features a rating hover-submenu (0-10) for rapid "Taste DNA" feedback.
    - **Live UI Sync**: Games are automatically removed from lists (e.g., Love, Free) if their status is changed to a "hidden" category.
- **Robust Header Image Rendering**:
    - Overhauled `GameHeaderImage` component with a multi-stage fallback (Shared Akamai -> Fastly -> Legacy -> Cloudflare).
    - Integrated support for metadata-hashed URLs to restore broken art for unreleased and new games (e.g., *Marathon*, *Keeper*).
    - Added `referrerPolicy="no-referrer"` to bypass ORB security blocks on newer Steam CDNs.
- **Verified Discovery Threshold**:
    - Implemented a mandatory **1-vote minimum** (Positive or Negative) for discovery lists (Love, Free, Tags) in `solve_user_taste.py`.
    - Ensures recommended games have existing user feedback while exempting **Upcoming** and **Backlog** lists to maintain visibility.

### Changed
- **Architectural Cleanup**:
    - Decoupled `GameStatus` types into `frontend/src/types.ts` to resolve Vite circular dependency warnings between Context and Components.
    - Standardized "Add Game" UI components across the application for keyboard-friendly UX.
- **Build System**: Incremented build version to **53**.

### Fixed
- **Catalogue Sorting**: Resolved "Sort to Top" priority logic where status-based sorting was incorrectly grouping items.
- **Solver Pipeline**: Fixed `AttributeError` in `get_base_filter_mask` by correcting numpy array handling in the discovery pass.

## [2026-03-05] - Taste DNA UI Overhaul & Strict Filtering
### Added
- **Taste DNA UI Redesign**:
    - Overhauled the "Personalization" results with a high-density two-column grid layout.
    - Expanded **"Games You'll Love"** and **"Backlog Priority"** lists from 10 to 30 items.
    - Integrated **"Top Free Games"** (10 items) and **"Games You'll Hate"** (10 items) as secondary discovery pillars.
    - Uniform styling for Love and Backlog lists, including rank numbering and predicted ratings.
- **Strict Free-to-Play Filtering**:
    - Implemented a **positive-indicator filter** for free recommendations.
    - Eliminated "Pharmageddon" false positives by strictly requiring the explicit `'Free to Play'` community tag.
    - Added automated exclusion of unreleased games ("Coming soon", "TBA") from discovery gems.

### Changed
- **Dynamic Schema Resilience**:
    - Backend (`server.py`) and Solver (`solve_user_taste.py`) now dynamically detect missing metadata columns (e.g., `tone_z`) and initialize them gracefully, preventing PyArrow load crashes.
- **Build System**: Incremented build version to **52**.

### Fixed
- **Distribution Integrity**: Resolved a pipeline bug where aggressive price regex was wiping out valid price data, restoring the model to its **20.44% OOS R2** peak.
- **Frontend Build**: Scrubbed unused imports and deprecated state variables to satisfy strict TypeScript compilation on the modern frontend.

## [2026-03-04] - The v7.1 Discovery Peak (20.44% R2)
... rest of changelog ...
