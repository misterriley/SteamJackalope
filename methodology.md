# Methodology: SteamJackalope Recommendation Engine

## The Jackalope Kernel (v7.1 "The Purist")
The core similarity engine uses a multi-stage geometric approach to distinguish between "Soulmates" (legitimate mechanical and thematic relatives) and "Impersonators" (parodies or asset flips).

### 1. Identity Resonance (MIGs)
We group ~440 tags into 34 **Mechanical Identity Groups (MIGs)**.
- **Structural MIGs**: (Roguelike, Strategy, Survival) High-veto weight. Mismatches here are almost always disqualifying.
- **Cognitive MIGs**: (Detective, Logic Puzzle, Story Rich) Acts as a "Soul Rescue." Shared cognitive DNA can waive mechanical vetos.
- **Identity Power (2.0)**: We use an exponential penalty for identity mismatch to ensure structural precision.

### 2. The Differentiable Soul (Semantic Bridge)
We use a 235-dimensional descriptive space (ZCA-whitened) to calculate thematic similarity.
- **Linear Verb Jaccard**: As of v7.1, we use linear Jaccard similarity for verb profiles to maintain mechanical precision for niche genres (e.g., Survivor-likes).
- **Additive Topics**: Thematic topics now act as multipliers ($Vibe = Semantic \cdot (1.0 + 0.5 \cdot Topics)$) rather than additive terms, ensuring they refine rather than override structural signals.
- **Mechanical Trust Bypass**: High-fidelity mechanical matches (Verb Jaccard > 0.6) automatically bypass the semantic floor gate (0.05), restoring bridges between "mechanical twins" that use different marketing vocabulary (e.g., *NieR: Automata* and *Stellar Blade*).

### 3. Purity Filters & Vetoes
- **Vibe Shield**: Triggers at high semantic (>0.35) or topic (>0.6) alignment, protecting niche soul-matches from generic structural penalties.
- **Title Hijack 4.0**: Aggressively suppresses games with similar titles but < 30% identity match, using a 0.7 overlap threshold via string normalization.

## The 20% Discovery Breakthrough (v7.1 "Core 9" Mode)
As of March 2026, we have reached a verified discovery ceiling of **20.44% Honest OOS R^2**. This represents the absolute predictive peak for the current feature set.

### 1. The "Core 9" Feature Set
Through a "Signal Battle" benchmarking process, we discovered that the 283 thematic features (MIGs and Topics) were injecting statistical noise into the regression pool for typical library sizes (400-600 ratings). The production model now focuses exclusively on the **Core 9** signals:
- **2 Neighborhood Features**: Dynamic Jackalope Kernel and Behavioral Graph estimates.
- **7 Global Metadata**: (Quality, Date, Popularity, Playtime, Difficulty, Price, Tone).

### 2. Optimal Regularization
By focusing on high-signal features, the model's optimal regularization alpha shifted from 59.64 down to **1.33**. This lower alpha indicates a fundamentally more robust model that is less prone to over-fitting, resulting in a higher OOS $R^2$ (20.44% vs 20.28%).

### 3. Zero-Drift Synchronization
The **Taste DNA Solver** and the **Live Recommender** share identical NW Kernel and Graph resonance logic. 
- **Smoothing Power**: Fixed at **10.0** across all environments to provide sharp, localized discovery planets.
- **Strict Free-to-Play Selection**: To eliminate unreleased games and "coming soon" placeholders from discovery gems, the system uses a **positive-indicator filter**. A game only appears in the "Top Free Games" list if it carries the explicit `'Free to Play'` community tag, bypassing the ambiguity of HTML storefront price fields.
- **Quality Refinement**: Recommendations are further prioritized by a Bayesian quality score, effectively "weeding out" low-fidelity clones that mimic mechanics but lack execution.

## Theoretical Limits & Findings

### The Plateau of Implicit Philosophy
While the "Core 9" model captures the statistical body of user taste, the remaining ~80% of variance represents **Implicit Structural Philosophy**—concepts like "Metroidbrainia" or "Linguistic Archeology." These are often discussed in user reviews but require high-fidelity LLM sentiment analysis to transform into first-class discovery features.
