# Methodology: SteamJackalope Recommendation Engine

## The Jackalope Kernel (v6.0 "The Oracle")
The core similarity engine uses a multi-stage geometric approach to distinguish between "Soulmates" (legitimate mechanical and thematic relatives) and "Impersonators" (parodies or asset flips).

### 1. Identity Resonance (MIGs)
We group ~440 tags into 34 **Mechanical Identity Groups (MIGs)**.
- **Structural MIGs**: (Roguelike, Strategy, Survival) High-veto weight (15.0). Mismatches here are almost always disqualifying.
- **Cognitive MIGs**: (Detective, Logic Puzzle, Story Rich) Acts as a "Soul Rescue." Shared cognitive DNA can waive mechanical vetos.
- **Semi-Structural**: (Shooter, RPG) Medium weight (6.0).

### 2. The Differentiable Soul (Semantic Bridge)
We use a 235-dimensional descriptive space (ZCA-whitened) to calculate thematic similarity.
- **Vibe Shield**: If a thematic match is in the top 1% (rank-based) AND absolute similarity > 0.12, structural vetos are disabled.
- **Contextual Floor**: The semantic floor (0.15) is waived if games share a **Structural MIG** or have > 85% mechanical identity.

### 3. Purity Filters & Vetoes
- **Perspective Veto**: Penalizes 2D/3D mismatches unless bypassed by the Cognitive Bridge or Vibe Shield.
- **Setting Clash**: Penalizes Sci-Fi vs Fantasy unless the vibe is top-tier.
- **Title Hijack 4.0**: Aggressively suppresses games with similar titles but < 30% identity match, with exemptions for "Remake", "Definitive", or "Enhanced" suffixes.

## Zero-Drift Pipeline
To ensure rating predictions remain consistent, the **Taste DNA Solver** and the **Live Recommender** share the identical `calculate_jackalope_kernel` function.
- **Solver**: Extracts ~45 structural feature weights via Ridge Regression.
- **Recommender**: Uses those weights in a linear dot-product with the unified kernel output.

## Theoretical Limits (v6.0 "The Oracle" Findings)

### The Plateau of Implicit Philosophy (82.19% Accuracy)
As of v6.0, we have reached a statistical ceiling for tag and text-based similarity. While **Jackalope-ULTRA** successfully identified the "Mechanical Body" and "Thematic Soul" for 82% of soulmate pairs, the remaining 18% represent cases of **Conceptual Innovation** that are invisible to metadata.

#### 1. The Body-Mind Unified Theory
To break the 70% barrier, we implemented **Rare MIG Resonance**.
- **The Body (MIGs)**: We trust mechanical resonance only when it is **rare** (e.g., *Detective, Logic, Translation*). Sharing a rare MIG is weighted 10x more heavily than sharing a common one (e.g., *Action, RPG*).
- **The Soul (Semantic Dimensions)**: We identified 38 **Cerebral Dimensions** (e.g., *Mystery, Logic, Ancient*) that act as an "Intellectual Bridge." 
- **The Filter (Trope Rejection)**: We successfully suppressed "SEO-Hijackers" (parodies like *Car Dealer Simulator*) by measuring their **Trope Loading**. If a game loads heavily on common filler tropes (NSFW, RPGMaker, Match 3), its semantic similarity to a high-fidelity seed is penalized.

#### 2. Structural Parity
We discovered that **MIG Count** (the total number of identity groups a game occupies) acts as a "Quality Floor." High-fidelity soulmates (e.g., *Chants of Sennaar* and *TUNIC*) share **Structural Complexity Parity** (both having 8+ MIGs), while low-effort mimics are often structurally shallow (< 4 MIGs).

#### 3. The Remaining 18% (Implicit Architecture)
The failure set consists of games connected by **Implicit Structural Philosophy**—concepts like "Metroidbrainia" or "Linguistic Archeology." These are discussed in user reviews but are not yet first-class features in the 235-dimensional semantic space.

## Next Phase: The 20% Discovery Breakthrough (v7.0 "All-In" Ridge)
As of March 2026, we have reached a new verified discovery ceiling of **20.28% Honest OOS R^2**. This represents a fundamental shift from pairwise similarity to a holistic, high-dimensional regression model.

### 1. The "All-In" Feature Matrix
The model now learns weights for a unified pool of **292 features**:
- **7 Global Metadata**: (Quality, Date, Popularity, Playtime, Difficulty, Price, Tone).
- **34 Structural MIGs**: Binary membership in mechanical identity groups.
- **249 Thematic Topics**: Projections into the 249-dimensional topic distribution.
- **2 Neighborhood Features**: Dynamic Jackalope Kernel and Behavioral Graph estimates.

### 2. Strict Ground Truth Policy
We discovered that unverified "import predictions" (bot guesses) were injecting significant noise into the training pool, capping discovery at ~9%. By strictly filtering for **verified human ratings** (`status == 'rated'`), we cleared the signal path, allowing the model to reach the 20% threshold.

### 3. High-Resolution Regularization
To maximize generalization, we implemented a manual 5-fold OOS assembly with a **50-point logarithmic alpha sweep** ($10^{-3}$ to $10^6$). This allowed the system to identify the precise discovery-optimal alpha (~59.64) where the model captures true human preference without over-fitting to the 481-sample training set.

### 4. Zero-Drift Synchronization
The 249 thematic weights learned by the solver are now first-class citizens in the backend. `app/server.py` correctly applies these coefficients to the topic distributions of all 100,000+ games in the library, ensuring the 20% discovery power is fully realized in production recommendations.
