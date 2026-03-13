# Tasklist

## 🎯 Active Tasks
- [ ] **Achievement-Based Engagement Profiling**: Research if achievement completion curves (drop-off vs. 100% completion) can act as a proxy for "Intellectual Depth."
- [ ] **Review Template Extraction**: Use the structured user review templates (e.g., "☑ Significant brain usage") as a first-class feature for high-fidelity ranking.
- [ ] Optimize Solver performance for 100k+ games (vectorize 2D Jaccard path further).
- [ ] Transition `more_like_this.json` to a sparse matrix format for production optimization.

## ✅ Recently Completed
- [x] **Solver Fix (March 2026)**: Fixed critical bug in `solve_user_taste.py` where scalar z-scoring was zeroing out all features; restored high-fidelity profile accuracy.
- [x] **Production Build Recovery**: Resolved 35+ TypeScript and ESLint errors blocking the Vite production build; synchronized development code with the `dist/` deployment folder.
- [x] **Visual Context & Interactive Media (v7.2)**:
    - Built a high-fidelity interactive hover card with an automated media slideshow (trailers/screenshots).
    - Implemented a magnitude-sorted, color-coded score breakdown chart for real-time model transparency.
    - Integrated a raw stat grid (Price, Difficulty, Length) directly into the recommendation previews.
    - Added disk-based media caching (`data/media_cache/`) to the backend to ensure low-latency hover response.
    - Implemented a manual **"Sync Data"** button in the Interactive view to bypass browser caching and force-reload fresh Taste DNA.
- [x] **High-Fidelity Similarity Function**: Built an optimized weighted similarity algorithm (Desc: 0.445, Verbs: 0.233, Tags: 0.174, Graph: 0.148) with strict quality/review filters to guarantee top-tier recommendations.
... [rest of recently completed] ...
