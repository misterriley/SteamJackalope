# Changelog

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
### Added
- **"Core 9" Discovery Mode**:
    - Reached a verified **20.44% Honest OOS R^2** ceiling by pruning 283 "noisy" features (MIGs and Topics) from the primary regression pool.
    - Focuses strictly on high-signal components: Kernel, Graph, Quality, Date, Popularity, Playtime, Difficulty, Price, and Tone.
    - Reduced optimal regularization ($\alpha$) from 59.64 to **1.33**, indicating a more robust and generalizable model.
- **Jackalope Kernel v7.1 ("The Purist")**:
    - **Mechanical Trust Bypass**: Restored critical discovery bridges (e.g., *NieR: Automata* -> *Stellar Blade*) by allowing high-fidelity mechanical matches (Verb Jaccard > 0.6) to bypass the semantic floor gate.
    - **Linear Verb Jaccard**: Reverted to linear Jaccard for verb profiles to maintain mechanical precision for niche genres like Survivor-likes.
    - **Additive Topics**: Thematic topics now act as multipliers ($Vibe = Semantic \cdot (1.0 + 0.5 \cdot Topics)$) to ensure structural signals remain dominant.
- **Rich Insight Synchronization**:
    - Restored **North Stars** (top 5), **Backlog Priority**, **Predictive Tags** (with "Tag Expert" gems), and **Hate Lists** to the Taste DNA profile.
    - Expanded **"Similar to Favorites"** to include discovery neighbors for all 9/10 rated seeds.
    - Implemented automatic **Seed Name Recovery** from master metadata to fix `nan` titles in user profiles.

### Changed
- **NW Smoothing Power**: Synchronized at **10.0** across Solver, Backend, and Research for sharper discovery planets.

### Fixed
- **Frontend Stability**: Fixed `TypeError: val.toFixed is not a function` on the Analyze page by filtering metadata loop for numeric values.
- **Data Quality**: Resolved "Dead Kernel" bug where aggressive squaring and gates were zeroing out mechanical similarities.

## [2026-03-04] - The 20% Discovery Breakthrough (v7.0)
### Added
- **"All-In" Ridge Discovery Model (v7.0)**:
    - Achieved a verified **20.28% Honest OOS R^2**, a ~200% relative improvement over the v4.2 baseline.
    - **Integrated Signal Pool**: Unified 292 features including Metadata, Behavioral Graph Similarity, 249 Thematic Topics, and Jackalope Kernel estimates.
    - **Strict Ground Truth Policy**: Training now purges unverified import predictions, using only honest human-assigned ratings (`status == 'rated'`).
    - **High-Resolution Alpha Sweep**: Implemented 50-point logarithmic sweep ($10^{-3}$ to $10^6$) to identify the discovery-optimal regularization peak ($\alpha \approx 59.64$).
- **Thematic Discovery Port**: Backend (`app/server.py`) now correctly applies the 249-dimensional topic weights during linear scoring.

### Changed
- **Leak-Proof CV**: Migrated from `LOOCV` to **Manual 5-Fold OOS Assembly**, ensuring neighborhood features for test games are derived strictly from training anchors.
- **Production Parity**: `pipeline/solve_user_taste.py` is now a literal mathematical port of the research breakthrough logic.

## [2026-03-04] - The God-Kernel Unification
### Added
- **Jackalope Kernel v6.0 ("The Oracle")**:
    - **Cognitive Bridge**: Allows shared intellectual DNA (Logic, Detective) to waive camera perspective vetos.
    - **Vibe Shield**: Top 1% thematic matches (Semantic CDF) now bypass all structural penalties.
    - **Title Hijack 4.0**: Surgical SEO-parody veto with suffix-awareness for legitimate Remakes/Remasters.
    - **Contextual Semantic Floor**: Dynamic semantic gates that trust the "Body" (Identity) if the match is structural or >85% pure.
- **Zero-Drift Unified Path**: `common/utils.py` now provides a single source of truth for both 1D (Recommender) and 2D (Taste DNA Solver) kernel calculations.

### Changed
- **MIG Taxonomy**: Reclassified MIGs into `STRUCTURAL`, `COGNITIVE`, and `SEMI_STRUCTURAL` for tiered weighting.
- **Identity Power**: Transitioned to dynamic identity power (1.0 to 3.0) based on structural matching.

### Fixed
- Fixed `KeyError` in kernel component extraction during solver loops.
- Fixed `IndentationError` in `common/utils.py` title-match logic.
- Resolved "Successor" failure modes for games like *Chants of Sennaar* -> *TUNIC*.
