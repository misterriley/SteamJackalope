# User Profile: misterriley (Steve Riley)

This document provides context about the user's working style, technical preferences, and expectations for AI agents working on the SteamJackalope project.

## 🛠️ Working Style & Environment
- **Environment**: Windows 11, VS Code with the Gemini/Cline extension.
- **Workflow**: Iterative and documentation-driven. Significant changes are tracked via Build numbers (Build 1, 2, 3...). 
- **Tooling**: Heavy use of PowerShell. Note that command chaining must use `;` instead of `&&`.
- **Git**: Uses Git LFS for large binary artifacts (`.npy`, `.parquet`). Commits should be "Why-focused" and summarize algorithmic or architectural shifts.

## 🧠 Technical Preferences
- **Statistical Rigor**: Steve is meticulous about mathematical consistency. He prefers explicit Bayesian smoothing, proper distribution centering, and robust transformations (CLR, Probit, PCA-ZCA Whitening).
- **Data Fidelity**: Values high-fidelity data collection. Prioritizes English-language review counts but maintains global coverage. Uses direct Storefront scraping for high-fidelity user tags.
- **Performance**: Optimizes for memory efficiency (mmap, float16 storage, float64 math) to ensure the system runs on standard cloud tiers.
- **Stability**: Build 15 introduced a "Unified Linear Scorer" mandate. All future changes must maintain 100% mathematical parity between the Taste DNA Solver and the Recommendation Engine.

## 🤝 Expectations for AI Agents
- **Proactiveness**: Agents are expected to identify root causes (e.g., missing stabilization files or misaligned indices) rather than just patching symptoms.
- **Verification**: Always verify changes with diagnostic scripts or existing tests. When in doubt, "push the tags through and back through the pipeline" to see the effect.
- **Documentation**: Keep `orientation.md`, `methodology.md`, and `CHANGELOG.md` updated. Surfacing "Internal Workings" in documentation is encouraged.
- **Communication**: Be direct and concise. Avoid fluff. Explain the "Why" behind proposed plans.

## 🚩 Critical User Mandates
- **Single Source of Truth**: `common/constants.py` must manage all magic numbers and paths.
- **Artifact Synchronization**: Row counts and ordering in `.npy` and `.parquet` files must ALWAYS match `data/pipeline_games_clean.csv`.
- **Tag Stabilization**: Always save tag names and prior anchors (`G_final`, `V_prior`) to prevent "Global Shift" instability.
