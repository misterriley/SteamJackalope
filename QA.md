# SteamJackalope Quality Assurance (QA) Test Suite

This document serves as the manual verification protocol for human testers and developers. Perform these tests before any major push or build increment.

## 🏗️ Core Navigation & UI
- [ ] **Splash Page Routing**: Click the "SteamJackalope" logo in the header.
    - *Expectation*: Redirects to the central Splash Page hub.
- [ ] **Hub Actions**: Click "Find Games", "Analyze My Catalogue", and "Explore Data" buttons on the Splash Page.
    - *Expectation*: Correctly routes to Recommender, DNA Solver, and Lists views respectively.
- [ ] **Tab Switching**: Click through Recommender, Lists, About, Methodology, and Changelog.
    - *Expectation*: Content updates instantly; active tab is highlighted.
- [ ] **Mobile Responsiveness**: Resize window to narrow width.
    - *Expectation*: Navigation items maintain `gap-2` spacing; layout wraps without breaking.
- [ ] **Changelog Sync**: Open the Changelog tab.
    - *Expectation*: Dynamic fetch succeeds; latest build number matches `build_count.json`.

## 🧪 Recommender Features
- [ ] **Autocomplete**: Type "Witcher" or "Cyber" in the seed search.
    - *Expectation*: Results appear; clicking one adds it to the seed list.
- [ ] **Surprise Me**: Click the "Surprise Me (Random)" button.
    - *Expectation*: Random prompt/seeds are selected; recommendations generate automatically.
- [ ] **Trending Random**: Click the flame icon (Trending).
    - *Expectation*: A currently popular Steam game is selected as a seed.
- [ ] **Discovery Slider**: Set Discovery to max ("Wild Cards") and Rating to max ("Loved").
    - *Expectation*: Results shift to high-rated games with < 500 reviews.
- [ ] **Price Slider**: Adjust the "Price" slider to max left ("Cheap").
    - *Expectation*: "Free" and low-cost games dominate the top results. Match score reflects the price contribution.
- [ ] **Weight Contributions**: View any game card in the recommender.
    - *Expectation*: The "Weight Contributions" debug bars are always visible. Ensure "Pr" (Price) is included in the list.
- [ ] **NSFW Blurring**: Toggle "Blur NSFW" ON.
    - *Expectation*: NSFW games appear but header images are blurred. Verify blur state persists between Personalized and Recommender views.
- [ ] **Profile Filtering**: Load a Taste DNA profile. Toggle the "Filter Profile Games" segmented control.
    - *Expectation*: 
        - **None**: Owned games appear in results.
        - **Rated**: Only games verify-rated in the UI are excluded.
        - **All**: Every game in the user's Steam library is excluded.

## 🏷️ Steam Integration (Build 13)
- [ ] **Card Meta-Links**: Click a Tag or Genre on a game card.
    - *Expectation*: Opens the validated Steam store link (e.g., `/tags/en/RPG`) in a new tab.
- [ ] **App Links**: Click the game title or external link icon.
    - *Expectation*: Opens the Steam App page.
- [ ] **Difficulty Tag Links**: In the Lists -> Difficulty Tags view, click a predictor tag.
    - *Expectation*: Opens the validated Steam store link for that tag.

## 📊 Data & Insights
- [ ] **Ranking Lists**: Cycle through Rating, Popularity, Playtime, and Age.
    - *Expectation*: Top/Bottom 50 tables populate with correct data types (pts, reviews, hours).
- [ ] **Live Re-ranking**: Adjust the Discovery slider while on the "Rating" list.
    - *Expectation*: The ranking updates in real-time.

## 📐 Documentation & Math
- [ ] **Math Rendering**: View the Methodology page.
    - *Expectation*: LaTeX formulas (e.g., $z = \frac{x - \mu}{\sigma}$) render as clean images/KaTeX, not raw text.

## 🧬 Personalization Engine (Build 15)
- [ ] **Data Acquisition**: Enter a valid SteamID64 and click "Start Analysis".
    - *Expectation*: Background task starts; UI transitions to "Acquiring Data" state.
- [ ] **Verification UI**: Review the predicted ratings table.
    - *Expectation*: Table is sortable; ratings can be manually adjusted; "Ignore" toggle works.
- [ ] **Taste Solving**: Click "Solve My Taste DNA".
    - *Expectation*: Pipeline completes; Insights dashboard displays Predictive Tags and Taste Anchors.
- [ ] **Unified Recommender**: Click "Apply to Recommender".
    - *Expectation*: Tab switches to Recommender; "PERSONALIZED" badge appears; top results match "Games You'll Love" list from Insights.
- [ ] **Semantic Slider Import**: After "Apply to Recommender", check the "Semantic Match" slider.
    - *Expectation*: The slider is automatically set to the user's solved semantic weight (e.g., ~0.90) instead of the default 1.0.
- [ ] **Key Vibe Dimensions**: Hover over a Vibe Dimension (✨ icon) in Insights.
    - *Expectation*: Displays a list of positive/negative keywords and a scatter plot of personal rating correlation.
- [ ] **Vibe Labeling**: Check Vibe Dimension titles.
    - *Expectation*: Titles use the composite word-sum format (e.g., "Word1 + Word2 vs. Word3 + Word4").
- [ ] **Hybrid Anchors**: View "Taste Anchors" in Insights.
    - *Expectation*: "North Stars" display an alignment percentage > 100% (calibrated hybrid score).
- [ ] **Predictive Multipliers**: Adjust a slider (e.g., Quality) in Personalized mode.
    - *Expectation*: Results re-rank based on the slider acting as a multiplier on the DNA weight.
- [ ] **Error Recovery**: Click the "Reset App" button in the header.
    - *Expectation*: Storage is cleared; page reloads; app returns to factory defaults.
