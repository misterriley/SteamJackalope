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

## 🖱️ Context Menu & Categorization (Build 53)
- [ ] **Right-Click Activation**: Right-click any game card in Recommender, Solve, or Catalogue.
    - *Expectation*: Custom context menu appears at mouse position.
- [ ] **Visual Indicators**: Open menu for a game already in "Backlog."
    - *Expectation*: A checkmark appears next to "Add to Backlog" and the item is highlighted.
- [ ] **Rating Submenu**: Hover over "Rate Game."
    - *Expectation*: A numeric 0-10 submenu appears. Clicking a rating updates the game's status to "Rated" and solves/refreshes as needed.
- [ ] **Live UI Removal**: Right-click a game in "Games You'll Love" and mark it as "Ignored."
    - *Expectation*: The game immediately disappears from the list without a full page reload.
- [ ] **Global Sync**: Categorize a game in the Recommender, then check the Catalogue.
    - *Expectation*: The new status is reflected in the Catalogue table.

## 🖼️ Art & Assets (Build 53)
- [ ] **Metadata Fallback**: Search for a new or unreleased game (e.g., *Marathon* or *Keeper*).
    - *Expectation*: Header image renders correctly using hashed metadata URLs, bypassing the legacy `header.jpg` 404s.
- [ ] **NSFW Blurring**: Toggle "Blur NSFW" ON.
    - *Expectation*: NSFW games appear but header images are blurred. Verify blur state persists between Personalized and Recommender views.
- [ ] **ORB Security Check**: View images on different browsers (Chrome/Firefox/Safari).
    - *Expectation*: Steam CDN images load without being blocked by `OpaqueResponseBlocking` thanks to `no-referrer` policy.

## 🧪 Recommender Features
- [ ] **Scroll to Top**: Scroll down at least 500px in the recommender results.
    - *Expectation*: A floating "Scroll to Top" button appears. Clicking it smoothly returns the view to the search bar.
- [ ] **Autocomplete**: Type "Witcher" or "Cyber" in the seed search.
    - *Expectation*: Results appear; clicking one adds it to the seed list.
- [ ] **Reset All (Profile-Aware)**: Load a Taste DNA profile, adjust sliders, and add a seed game. Click "Reset All".
    - *Expectation*: 
        - **Profile Active**: Sliders snap back to the solved DNA weights; seed games, prompt, and tags are cleared; "PERSONALIZED" badge remains.
        - **No Profile**: All sliders and inputs return to hardcoded global defaults.
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
- [ ] **Profile Filtering**: Load a Taste DNA profile. Toggle the "Filter Profile Games" segmented control.
    - *Expectation*: 
        - **None**: Owned games appear in results.
        - **Rated**: Only games verify-rated in the UI are excluded.
        - **All**: Every game in the user's Steam library is excluded.

## 🏷️ Steam Integration
- [ ] **Card Meta-Links**: Click a Tag or Genre on a game card.
    - *Expectation*: Opens the validated Steam store link (e.g., `/tags/en/RPG`) in a new tab.
- [ ] **App Links**: Click the game title or external link icon.
    - *Expectation*: Opens the Steam App page.

## 📚 Catalogue Management
- [ ] **Catalogue Loading**: Navigate to the "Catalogue" view.
    - *Expectation*: System fetches library data; "My Game Catalogue" table appears.
- [ ] **Categorical Sorting**: Click the "Status" column header.
    - *Expectation*: Games group by Rated -> Played -> Backlog -> Wishlist -> Ignored -> None.
- [ ] **Infinite Scroll Performance**: Scroll quickly through a 1000+ game library.
    - *Expectation*: Rendering remains fluid (no lag); "Back to Top" button appears and works.
- [ ] **Real-time Sync**: Change a game's status (e.g., Backlog -> Played).
    - *Expectation*: A "Saving..." notification appears; status persists after a page refresh.
- [ ] **Auto-Promotion**: In Catalogue or Verify view, move a 0-rating slider to 7.
    - *Expectation*: Game status automatically promotes to "rated" and "Ignore" is unchecked.

## ⌨️ Keyboard & Input
- [ ] **Arrow Navigation**: Type "Elden" in the manual add search. Use Arrow Keys.
    - *Expectation*: Active selection highlights and scrolls into view within the dropdown.
- [ ] **Keyboard Commit**: Press Enter on a highlighted search result.
    - *Expectation*: Game is added to the list; dropdown closes; input clears.
- [ ] **Manual Default Logic**: Add a game manually.
    - *Expectation*: Game defaults to "Backlog" status, rating "5", and "Ignore: Checked".

## ⚙️ Kernel & Identity Enforcement
- [ ] **Symmetric Perspective Veto**: Use "Disco Elysium" (Isometric) as a seed. Search results for "Leons Identität" (First-Person) or "Portal 2" (First-Person).
    - *Expectation*: Perspective-clashing games are heavily penalized ($0.001x$) and should not appear in the top 100 results, regardless of vibe similarity.
- [ ] **RPG Anchor Enforcement**: Use "Mass Effect (2007)" (RPG) as a seed. Search for "Front Mission Evolved" (Shooter, No RPG).
    - *Expectation*: Pure action games without the "RPG" tag are vetoed ($0.001x$) when matching an RPG seed.

## 📐 Documentation & Math
- [ ] **Math Rendering**: View the Methodology page.
    - *Expectation*: LaTeX formulas (e.g., $z = \frac{x - \mu}{\sigma}$) render as clean images/KaTeX, not raw text.

## 🧬 Personalization Engine (Build 53)
- [ ] **Data Acquisition**: Enter a valid SteamID64 and click "Start Analysis".
    - *Expectation*: Background task starts; UI transitions to "Acquiring Data" state.
- [ ] **Verification UI**: Review the predicted ratings table.
    - *Expectation*: Table is sortable; ratings can be manually adjusted; "Ignore" toggle works.
- [ ] **Taste Solving**: Click "Solve My Taste DNA".
    - *Expectation*: Pipeline completes; Insights dashboard displays Predictive Tags and Taste Anchors.
- [ ] **Discovery Vote Threshold**: Solve a profile and check "Games You'll Love" or "Top Free Games."
    - *Expectation*: No games with 0 total reviews are recommended in these discovery lists.
- [ ] **Backlog/Upcoming Immunity**: Check "Backlog Priority" and "Upcoming Games."
    - *Expectation*: Niche or unreleased games with 0 reviews ARE allowed to appear here.
- [ ] **Unified Recommender**: Click "Apply to Recommender".
    - *Expectation*: Tab switches to Recommender; "PERSONALIZED" badge appears; top results match "Games You'll Love" list from Insights.
- [ ] **Key Vibe Dimensions**: Hover over a Vibe Dimension (✨ icon) in Insights.
    - *Expectation*: Displays a list of positive/negative keywords and a scatter plot of personal rating correlation.
