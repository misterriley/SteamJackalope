# Changelog

## [2026-03-04] - The 20% Discovery Breakthrough
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
