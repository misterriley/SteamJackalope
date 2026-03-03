# Methodology

## 1. Data Collection & Preprocessing
The initial dataset consists of Steam games as of March 2025 downloaded from [Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset). Each month Steam is scraped for new content/scores/reviews/descriptions/comments etc. Data is collected directly from the [**Steam Storefront**](https://store.steampowered.com/) and the official [**Steam API**](https://developer.valvesoftware.com/wiki/Steam_Web_API) to ensure high fidelity for user tags and review counts. Review counts are prioritized from **English** language sources where available, falling back to global counts to ensure relevance for English-speaking users while maintaining coverage for international titles.

## 2. Semantic Embeddings
To enable natural language search, we use the [**all-mpnet-base-v2**](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) SentenceTransformer model. This model provides superior qualitative and thematic matching compared to smaller models, utilizing 768-dimensional vectors to capture deep narrative nuance.

- **Descriptive Path:** We focus our semantic matching on the narrative and "vibe" of a game using its **Short Description** and **User Reviews**. This captures "how it feels" and "how it's talked about" by real players. We have explicitly decoupled categorical properties (Genres/Tags) from the semantic slider to ensure that natural language searches prioritize qualitative vibes rather than redundant categorical matches.

- **Process:** Each piece of text is converted into a high-dimensional vector that represents its meaning. To ensure semantic consistency and eliminate "style-noise," all input text (descriptions, reviews, genres, and tags) is **lowercased** before encoding. Groups of words have their vectors pooled, allowing long text to be represented by a single vector. To optimize performance, we utilize **GPU Acceleration (CUDA)** on compatible NVIDIA hardware, reducing the processing time for our ~150,000 game database from hours to minutes.

- **Usage:** The similarity between these vectors is measured using pure [dot product](https://en.wikipedia.org/wiki/Dot_product) in a unit-normalized space. This allows the model to recognize synonyms like "peaceful" and "calm" as closely related concepts. When using "Seed" games, the system averages the descriptive vectors of the seeds to find games with a similar narrative profile.

- **Uncentered ZCA Whitening:** To improve semantic precision, we apply **Uncentered Zero-phase Component Analysis (ZCA) Whitening** to the vectors. Unlike standard whitening, we do not subtract the mean; this preserves the origin of the vector space, ensuring that "empty" or "neutral" games remain at the zero-point. This process decorrelates dimensions and magnifies unique, salient characteristics of a game (similar to a TF-IDF effect for vectors), ensuring that specific narrative themes are not drowned out by common "Steam-speak" noise. We retain 95% of the global variance, which typically compresses the 768-dimensional space into approximately 235 principal directions.

- **Semantic Variance Parity:** Semantic embeddings naturally exhibit lower variance (~11x less) than whitened tag vectors. To ensure that the **LASSO Regression** treats "Vibe" and "Tags" as mathematically equal features, we apply a **Global Scaling Factor** (derived dynamically by the pipeline, currently **11.25x**) to the semantic dimensions. This prevents the model's sparsity penalty from prematurely dropping descriptive signals in favor of categorical tags.

- **Descriptive Dimension Labeling:** To make the 235 abstract ZCA dimensions human-readable, we use a **Composite Word Calibration** method. We extract the 10,000 most frequent single words from the dataset, project them into the whitened space, and find the **Vector Sum** of pairs that best align with each end of every axis. This results in descriptive labels like *"Exploration + Terraform vs. Gunplay + Ricochet"*, providing a high-contrast conceptual map of the latent space.

## 3. Tag Embeddings
Steam tags are user-contributed and often noisy. We apply a pipeline to transform these raw counts into robust "tag vectors" that capture the stylistic profile of each game.

- **Iterative EM Imputation:** To handle unobserved tags (Steam limits API responses to the top 20 tags), we use an Iterative Expectation-Maximization (EM) algorithm. The E-step imputes counts for unobserved tags using the current covariance matrix (conditional expectation), capped by the count of the 20th tag. The M-step recalculates the global mean and covariance from these augmented profiles. This is repeated until the model converges to a stable estimate of the missing data. The result is a best guess for the distribution of censored tags that would fall outside of the top 20. 

- **Stochastic Path Optimization:** To determine the optimal regularization constant $K$, we employ a simulation study. We select "reliable" games (with >1,000 votes) and simulate smaller versions of them by drawing samples from the data associated with one game at a time; the number of samples per game is a random variable taken from the actual distribution of sizes from the full dataset. We then solve for the $K$ that minimizes the Sum of Squared Errors (SSE) between the regularized synthetic profiles and the true reliable profiles.

- **Bayesian Regularization:** We apply Bayesian smoothing using the optimized $K$ to shrink low-information games toward the global prior.

- **Swappable Transformations & Bayesian Regularization:** We support three transformation options to stabilize the variance of tag counts: the [Anscombe Transform](https://en.wikipedia.org/wiki/Anscombe_transform) ($2\sqrt{x + 3/8}$), [Centered Log-Ratio (CLR)](https://en.wikipedia.org/wiki/Compositional_data#Center_log_ratio_transform), or an Identity transform. Currently we are evaluating whether any of these outperforms the others for quality of matches. It appears that CLR is the best metric for capturing the geometric and informational properties of the space of tags, but more study is needed.

- **Process:** We first apply Bayesian regularization directly to the raw tag counts: $\text{Profile} = (C + K \cdot G) / (N + K)$, where $C$ is the count vector, $G$ is the global prior, and $N$ is the total votes. This ensures that the regularization happens on the "natural" scale of the data. We then apply the chosen transformation to both the regularized profile and the global prior. The final embedding is the difference between these two in the transformed space: $V = \text{Transform}(\text{Profile}) - \text{Transform}(G)$. This ensures that the "regularizing point" (a game with zero observed tags) resides at the origin ($0$) of the embedding space, making it a neutral reference for similarity calculations.

- **Tag Vectors as Regression Coefficients:** We use tag vectors that are normalized with a shrinkage coefficient as regression coefficients in the model, such that the contribution of a set of tags for a game in the data set is the dot product of its tag vector by a seed game's shrunken normalized vector. This effectively penalizes similarity scores for games with low-information (short) tag vectors, ensuring that recommendations are based on strong, confident tag matches. We assume that short vectors are dominated by Gaussian-distributed noise, and so we estimate this noise with a parameter $\lambda$ that is determined by fitting a [Chi-distribution](https://en.wikipedia.org/wiki/Chi-distribution) to the lengths of low-tag vectors. We set $\lambda$ to the 95th percentile of this fitted distribution, which represents the "noise floor" of the embedding space. This ensures that a vector's length must be statistically significant before it can achieve high similarity scores.

- **Whitening & Dimensionality Reduction:** To ensure stability and remove numerical noise, tag vectors undergo **Truncated PCA-ZCA Whitening**. The data is projected onto its top principal components, which capture 95% of the global variance while eliminating singular dimensions (such as the linear constraint imposed by the CLR transform). This process decorrelates the tags and prevents "ghost" similarities between unrelated games.

## 4. Personalization: Taste DNA
To provide bit-perfect recommendations tailored to individual history, we use a supervised regression pipeline to build a "Taste DNA" profile.

- **Soft-Labeling:** We estimate a user's potential rating for every game in their library using a combination of their explicit Steam reviews and their relative playtime. We apply a **Log-Normal Kernel** to the global playtime distribution of each game to predict the probability of a positive review at the user's specific playtime ($p+(t)$). This probability is blended with the global quality score to produce a predicted rating on a 0-10 scale.

- **User Ratings:** Soft labels are only guesses, so an intermediate step is to have the user correct these guesses with their true ratings. 

- **LASSO Regression:** We solve for the user's "Taste DNA" using [**LASSO Regression**](https://en.wikipedia.org/wiki/Lasso_(statistics)). This technique is chosen for its ability to perform automatic feature selection; by applying an L1 penalty, the model identifies the specific subset of tags and metadata that truly drive a user's preferences, setting irrelevant features to exactly zero. In internal tests, L1 penalization outperforms L2 penalization in terms of generalization capability.

- **Adaptive Dimensionality:** To prevent overfitting while maintaining high fidelity, we use an **Adaptive Dimensionality** scheme for the tag space. The number of principal components $K$ used during solving is dynamically set based on the user's library size $N$, typically following $K = \text{clip}(N-6, 1, 243)$. This ensures that users with small libraries receive stable, broad profiles, while power users with hundreds of rated games receive highly granular, high-dimensional taste vectors.

- **Importing to Recommender:** Outputs from the **Taste DNA Solver** can be ported back into the recommender and users can adjust settings, add prompts, filter by tag/genre, or set seed games to adapt the outputs.

## 5. Quality Scoring
Instead of raw review percentages, we use a Bayesian score that smooths out noisy reviews. This allows the system to distinguish between a game with 10 positive reviews out of 10 and a masterpiece with 98,000 positive reviews out of 100,000. 

The model we use is based around the idea that there is a latent quality score $Q$ for each game, a normally distributed range of possible experiences that users might have with that game, and a threshold where, if a user has an experience greater than the threshold, they will give a positive review. Under some minimal assumptions, this gives the relation $\frac{p}{p + n} = \Phi(Q)$, where $p$ and $n$ are the number of positive and negative reviews left for the game, respectively, and $\Phi$ is the [cumulative distribution function of a standard normal variable](https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function). However, we do not necessarily trust the overall review score, especially when there are only a few reviews; this is due to [the cold start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)). So, we introduce a variable $s$ that "shrinks" the percentage of positive reviews toward a group mean, where $s$ has a larger effect for games with fewer reviews. The variable $s$ can be thought of as a set of initial reviews shared by all games that reduces the variability of estimates of $Q$ when there are few reviews. The resulting regularization scheme yields the following formula: $\frac{p + s \cdot a}{p + n + s} = \Phi(Q)$; solving for $Q$ gives $Q = \Phi^{-1}\left(\frac{p + s \cdot a}{p + n + s}\right)$.  

- **Components:** $p$ and $n$ are positive and negative reviews, $a$ is the global positive review rate derived from the dataset, and $s$ is a regularization constant.

- **Tunable Discovery:** The "Discovery" slider allows users to adjust $s$ in real-time. A high discovery setting ("Wild Cards") uses a low $s$, allowing "hidden gems" with few reviews to climb the rankings, while a low discovery setting ("Known Quantities") uses a high $s$ to ensure only established, highly-reviewed titles reach the top. The application uses a precalculated grid (21 steps) to provide smooth transitions between these settings.

- **Probit Function:** $\Phi^{-1}$ is the [probit function](https://en.wikipedia.org/wiki/Probit), which converts probabilities into a linear scale (z-scores), making the differences between "great" and "perfect", or "bad" and "horrendous", more meaningful.

- **Score Normalization:** To maintain a consistent spread of scores in recommended games across different Discovery settings, we apply a linear transformation to the generated z-scores. This ensures that the mean scores of the top 500 and bottom 500 games remain aligned with a baseline distribution (s = 2000), preventing the score distribution from collapsing or expanding excessively at extreme parameter values.

- **Computational Efficiency:** To minimize memory usage on resource-constrained environments, the recommendation engine utilizes [Reduced Precision Arithmetic](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) (FP16) for large-scale vector operations and [Memory-Mapped I/O](https://en.wikipedia.org/wiki/Memory-mapped_file) for static data artifacts. This allows the system to perform high-dimensional similarity searches efficiently.

## 6. Playtime Regularization
To effectively rank games by length, we analyze the distribution of playtimes that are available as part of user reviews. We assume that negative reviews will bias the estimate of game length downward due to quitting the game before finishing it, so we discard those values. Median playtime estimates can be highly variable for games with few reviews, and to address this, we apply a Bayesian shrinkage method similar to our tag vector approach.

- **Stochastic Path Analysis:** We determine an optimal regularization constant $C$ by creating synthetic "small" games using reviews from "reliable" games (those with $\ge 80$ positive reviews). We solve for the $C$ that minimizes the error between the regularized estimate of the small game and the true median of the original reliable game.

- **Metric:** We regularize the logarithm of the median playtime: $\log(1 + \text{median})$.

- **Formula:** $\text{Regularized Log-Median} = \frac{n \cdot \text{Sample Log-Median} + C \cdot \text{Global Mean Log-Median}}{n + C}$, where $n$ is the number of positive reviews. This ensures that games with very few reviews are strongly pulled toward the global average, while games with substantial feedback retain their specific playtime identity.

- **Display:** The exponential of this regularized value, minus 1 and converted to hours, is displayed as the estimation of length. 

## 7. Difficulty Prediction (Build 68)
Steam games do not natively have a "Difficulty" rating. To solve this, we built a predictive model using external data from GameFAQs (~2,900 matched titles) mapped to Steam game features.

- **Model:** We train a **LASSO Regression** model (L1 Regularization) to predict difficulty using a combination of **Steam Tags** and **BERTopic Gameplay Topics**.

- **Robust Feature Transformation (Rank-INT):** To ensure robust generalization and prevent "niche bias" (where features rare in the training set but common in the general population, such as specific Sudoku topics, cause extreme over-prediction), the model utilizes **Rank-Based Inverse Normal Transformation (Rank-INT)** on all input features. Features are ranked across the **full 155,000 game population** and mapped to a normal distribution. This caps the influence of extreme outliers and prevents the linear model from over-extrapolating in high-density feature regions.

- **Tag Preprocessing (5 Iteration EM):** We use **5 Iterations** of our Expectation-Maximization (EM) imputation algorithm to recover "censored" tags. These counts are transformed using the **Centered Log-Ratio (CLR)** transform and shrunken via **Bayesian Regularization** ($K \approx 72.5$). 

- **Predictors:** The model identifies "Difficult", "Chess", and "Intentionally Awkward Controls" as top difficulty predictors. High-challenge gameplay topics like Topic 37 (Precision Platforming) are strongly weighted, while Topic 118 (Zen/Relaxing) is a top ease predictor.

- **Estimation & Display:** The model is applied to all ~155,000 games. Results are clamped between **0.0 and 10.0**, with a global mean of approximately **5.0**. These estimates are converted to **Global Z-scores** for the recommendation engine, ensuring parity with other metadata signals.

## 8. Content Sensitivity (NSFW Blur)
To maintain a safe and professional discovery experience, Steam Jackalope employs an **"NSFW Blur" Architecture**. 

- **Flagging:** Games are automatically flagged as `is_nsfw` based on a the html from the scraped page (where a big "Adult Only" sign will be) or the "Hentai" tag.  

- **UI Interaction:** The frontend uses a global **Blur NSFW**. When active, any game flagged as NSFW has its header image blurred using a CSS backdrop filter. This maintains the integrity of the recommendation list while giving users control over their visual environment.

## 9. The Jackalope

The final recommendation list is generated as a hybrid, blending normalized components:

1.  **Semantic Match:** Similarity between the natural language prompt and game text. This component is Z-scored to maintain a consistent scale relative to other features.

2.  **Tag Match:** Similarity between the latent tag vectors. In Build 16, this was unified with the Linear Scorer model. Every game $V$ used as a seed provides a set of regression coefficients $\beta_{seed} = V / (\|V\| + \lambda)$. The contribution is calculated as the penalized dot product $Score = (U \cdot \beta_{seed} / (\|U\| + \lambda)) \cdot 11.283$. This component is no longer Z-scored, as it operates on an absolute 0-10 rating scale calibrated to the user's taste.

- **Softmin Multi-Signal Blending**: When multiple similarity signals are active (e.g., a combination of Taste DNA, text prompts, and multiple seed games), the system employs a **Softmin-based blending** approach ($T=3.0$). Unlike simple additive averages which can be dominated by a single strong match, Softmin rewards "consensus"—games that align well with *all* active targets. This ensures that a multi-seed search for "Gothic Horror" + "Deckbuilder" correctly identifies titles that bridge both genres rather than just suggesting popular games from one or the other.

3.  **Quality Score:** Preference for loved vs. hated games, smoothed by the Discovery setting.

4.  **Popularity:** Preference for high vs. low player counts.

5.  **Game Age:** Preference for new vs. classic titles.

6.  **Length:** Preference for short vs. long experiences.

7.  **Difficulty Rating:** Preference for Easy vs. Difficult games.

8.  **Price Sensitivity:** Preference for free/budget vs. premium titles.

Users can tune these weights in the UI to prioritize long, easy, niche hidden gems or popular classics that are similar to a game of choice. 

**Data Reliability & Quality Adjustments:**

- **Price Normalization:** Steam prices follow a zero-inflated lognormal distribution. We apply a **Log-Transform** ($\ln(1 + \text{price})$) before calculating Z-scores. This makes the price slider psychologically linear (treating a doubling of price as a consistent step) and prevents expensive outliers from dominating the model. Games without a listed price are assigned a neutral Z-score of **0.0**, ensuring they are treated as "average information" rather than accidentally being flagged as free or expensive.

- **Review Count Repair:** To handle stale metadata, the system automatically repairs global review counts using raw individual reviews if our scrape finds more data than the storefront summary.

- **Age Stability:** Future release dates are clamped to the current date, and unknown dates are assigned a neutral z-score (0) to maintain distribution stability.

- **Global Clamping:** To prevent extreme outliers from disproportionately influencing the final score, all component z-scores are clamped between -8.0 and 8.0.

## 8. Filtering
The system allows real-time filtering for **Genres**, **Tags**, VR-Only titles, English language support, Software/Utilities (Breadcrumb detection), and Unreleased games to ensure the results are relevant to the user. The Genre filter supports multi-selection, ensuring that recommended games match at least one of the selected categories. Downloadable Content (DLC) is excluded from the database to focus recommendations on standalone games.

- **Robust Release Date Filtering**: To handle "Coming Soon" titles with stale metadata, the system uses a dual-layer filter. It compares the `parsed_date` against the build-time of the metadata file and performs an explicit text-search for placeholders like "Coming soon", "TBD", and "To be announced". This ensures unreleased games are correctly excluded even if their placeholder dates have technically passed.

## 15. Tag Dimension Explainability (Build 41)

The PCA-ZCA whitening process is excellent for decorrelating features and improving model performance, but it results in latent dimensions that are not immediately human-readable. To provide transparency into what these abstract dimensions represent, we introduced an explainability layer.

- **Analysis of Whitening Matrix**: We analyze the whitening transformation matrix (`w_tag.npy`), which contains the weights of the original tags for each principal component. For each dimension (column), we identify the top positive and top negative tags based on their weights.
- **Generating Descriptions**: A script (`research/analyze_tag_dimensions.py`) uses these influential tags to generate a concise, human-readable label that captures the essence of the dimension's axis (e.g., "Tactical Combat vs Adult Themes"). This gives qualitative meaning to the mathematical constructs.
- **UI Integration**: These descriptions are served via an API endpoint and displayed in the personalization view, allowing users to understand not just that they have a preference for "Dimension 5," but that "Dimension 5" represents a preference for "Atmospheric Mystery over Casual Adventure."

## 9. Insights & Rankings
The "Lists" page provides a curated view of the Steam library's extremes, including the highest and lowest quality games, the longest and shortest experiences, and predicted difficulty rankings. These lists utilize the same Bayesian models and predictive analytics used in the recommendation engine, providing transparency into how different games are positioned within our statistical model.

## 10. Playtime-Sentiment Correlation
To understand how player engagement relates to satisfaction, we utilize a **Kernel Smoothing** model to analyze the relationship between playtime and review sentiment (positive vs. negative). This allows us to estimate the probability that a player will enjoy a game based on how long they have played it.

- **Lognormal Kernel:** We use a Gaussian kernel in log-space to measure similarity between playtimes. This accounts for the fact that the difference between 1 hour and 2 hours is more significant than the difference between 100 and 101 hours.

- **Leave-One-Out Prediction:** For each review in a game's history, we estimate its probability of being positive by taking a weighted vote of all *other* reviews for that game. Reviews from players with similar playtimes carry the most weight.

- **Bayesian Regularization:** To handle timeframes with few reviews, we apply Bayesian smoothing, pulling the predicted probability toward the global average positive review rate (0.80).

- **Optimization:** The model's smoothing bandwidth ($\gamma$) and regularization strength ($s$) were globally optimized using a parallelized grid search over the entire dataset of 36,000+ games. We maximized the total log-likelihood (cross-entropy) of the historical review data, ensuring that every review across all games contributes equally to the calibration. 

- **Usage:** This model reveals patterns such as "early quitters" (negative reviews at low playtime) vs. "burnout" (negative reviews at extremely high playtime), providing deeper insights into the player experience beyond a single aggregate score. The smoothing bandwidth and regularization strength are used for estimating the probability of a positive review for the user given their playtime. 

## 11. Visual & UI Stability
To provide a polished and safe user experience, the modern frontend implements several automated visual treatments:

- **Grid Rendering Stability:** The recommendation grid utilizes persistent React keys based on game AppIDs and standardized CSS layout principles (avoiding flexbox on global containers) to prevent "Single Card" collapse bugs and ensure consistent card sizing across all screen resolutions.

- **Dynamic Tag Visualization:** Each game card dynamically renders its most relevant genres and top user tags, highlighting matches that align with the user's search intent or seed games.

## 12. Personalized Linear Scorer (Build 15)

To ensure perfect alignment between a user's library analysis and their discovery feed, Build 15 introduced a **Unified Linear Scorer** architecture. This replaces the "hybrid approximation" with a direct execution model.

- **Unified Feature Space:** Both the Taste DNA Solver and the Recommendation Engine utilize a standardized global feature space. Metadata (Age, Quality, Popularity, Price, etc.) are represented as **Global Z-scores**, while Steam tags are transformed using **Penalized Normalization** ($v / (\|v\| + \lambda)$) and then scaled by a global constant (**11.283x**) to match the variance of the Z-scored metadata.

- **Portable Beta Weights:** By removing user-specific scaling, the DNA Solver learns **Beta Weights** that are directly portable. These weights represent the absolute importance of each feature (e.g., "how many rating points is one unit of Difficulty worth to this specific user?").

- **Direct Execution:** When a Taste Profile is applied, the Recommendation Engine switches into **Linear Mode**. Instead of using internal biases, it calculates the final Match Score by directly running the user's solved regression model: $\text{Match Score} = (\text{Intercept} + \sum (\beta_i \cdot x_i)) / 3.0$. 

- **Absolute Sliders:** In this mode, the UI sliders represent the **Absolute Weights** of the solved DNA profile. A user can directly see and adjust the solved coefficients (e.g., "Quality weight = 2.5"). Moving a slider updates the profile's baseline in real-time, allowing for intuitive and transparent fine-tuning of the "Taste DNA."

- **Predicted Ratings:** Because the math is unified and utilizes a global scaling factor (3.0), the "Match Score" displayed on game cards becomes a calibrated **Predicted 0-10 Rating** for that game. This ensures bit-perfect parity between the "Games You'll Love" list in the analyzer and the Recommender results.

- **Backlog Discovery**: The system distinguishes between games in the user's library that have been "Completed" (playtime > 0 or manually rated) and those that are in the "Backlog" (playtime = 0 or ignored). Backlog games are eligible for discovery in specialized lists, allowing users to surface high-rated titles they already own but haven't yet experienced.

- **Numerical Stability:** To prevent precision-based ranking swaps, the unified scoring path utilizes explicit **float32** casting for all vector operations and tag normalization.

## 13. Robust Personalization (Build 41)

As the personalization engine matured, we introduced several features to ensure that the "Taste DNA" remains both statistically sound and human-readable.

- **Sample Bias Mitigation:** We identified that local scaling (using a user's library as the scaling population) introduced significant bias, over-representing low-variance features that weren't globally predictive. To solve this, the Taste DNA solver now operates on **Globally Standardized Features**, utilizing the true population variance as the anchor. This improved model generalization by ~4% in cross-validation tests.

- **Adaptive DNA Dimensionality:** To prevent the model from "overfitting" or memorizing small libraries, the solver dynamically scales the complexity of the tag space based on the number of ratings provided by the user. We use a smooth linear relationship: $K = \text{clamp}(40 + 0.7 \times N_{\text{ratings}}, \text{min}=40, \text{max}=243)$. This ensures that new users with few ratings are modeled using only broad, high-certainty genre components, while power users with hundreds of ratings gain access to high-fidelity, niche stylistic details.

- **Support-Based Tag Filtering:** High-dimensional models can sometimes create "phantom" associations (statistical aliasing) where a tag you haven't actually played shows up in your DNA because it is correlated with something you like. To solve this, we implemented a **Sanity Check** layer. A tag is only eligible to be displayed in the "Love/Hate" lists if it has direct evidence in your library (i.e., it appears in at least one game you have rated). This ensures that the DNA view is always grounded in your actual play history.

- **Taste Anchors (North Stars & The Abyss):** To identify the games that most purely embody or oppose a user's preferences, we calculate a **Hybrid Alignment** score. This sums the weighted contributions of both the Tag Vibe Vector and the Semantic Vibe Vector for every game in the database, ignoring metadata like quality or price. The result surfaces games that are the "mathematical center" of a user's qualitative and categorical preferences.

## 14. Explainability & Visualization (Build 41)

To provide transparency into the "why" behind recommendations, we introduced a suite of visual analytics in the personalization insights view:

- **Interactive Correlation Charts:** Users can hover over individual Taste DNA weights to see a scatter plot relating their ratings to that specific feature. This visualizes the raw evidence the model used to solve for a specific preference.

- **Logarithmic Scaling:** Features that follow power-law distributions (Popularity, Price, Length) are visualized using logarithmic X-axes. This provides a clear, undistorted view of relationships across orders of magnitude.

- **Discovery Optimization Scan:** A specialized bar chart visualizes the full $R^2$ correlation scan performed by the solver. This shows how your profile's fit improves or degrades across different "Wild Card" settings, making the selection of your optimal Discovery preference transparent.

- **Dynamic Data Cleaning:** The visualization layer automatically filters anomalous data (like future placeholder dates or year-zero entries) and handles non-positive values on log scales to ensure the charts remain high-fidelity and informative.

## 16. Transparency & Parity Refinement (Build 45)

To resolve subtle "slippage" between the analyzer's preview and the live recommender, Build 45 introduced a series of mathematical constraints to ensure **Perfect Ordinal Parity**.

- **The Neutral Anchor (5.0):** Both the solver and the recommender now utilize a fixed **5.0 intercept** as the "Average Game" baseline. This represents the prior expectation of a game's rating before any specific metadata or tag affinities are applied. By anchoring to 5.0, the system ensures that "Match %" calculations are centered on a meaningful, stable point.

- **Transparency Mode (Absolute Weights):** In Taste DNA mode, the UI sliders (Quality, Age, Tag Match, etc.) now represent the **Absolute Weights** learned by the solver. Previously, sliders acted as multipliers for hidden internal weights, which caused a "Squaring Bug" where weights were being multiplied by themselves upon import. By switching to absolute weights, the UI becomes a transparent reflection of the underlying model, ensuring that a "Quality" setting of 0.86 in the UI is exactly what the backend uses for scoring.

- **Intercept Isolation:** We identified a "Scaling Regression" (the "82% Bug") where high-affinity games were capped at an 82% match. This was caused by dividing the 5.0 intercept by the global scaling factor. In Build 45, we isolated the intercept: `Score = (Weighted_Sum / 3.0) + 5.0`. This preserves the 5.0 neutral point while allowing strong positive deviations to reach their true potential (e.g., 98-99% matches).

- **Discovery-Aware Bias Correction:** Because semantic and tag similarity features often have non-zero global means, an "Average Game" might accidentally receive a non-neutral score. To correct this, the system calculates the **Expected Score of a Random Game** for the current weights and subtracts this bias from the final result. This ensures that a game with no specific affinity to the user's profile correctly results in exactly a 5.0 score (50% match).

## 17. Consensus & Discovery Refinement (Build 46)

Build 46 refined the discovery experience by enhancing the recommendation blending logic and expanding the scope of library analysis.

- **Softmin Multi-Signal Blending**: The transition to Softmin blending (T=3.0) for multi-comparator signals (DNA + Seeds + Prompts) ensures that recommendations align with the **consensus** of all active targets. This prevents a single dominant signal (like a very specific prompt) from drowning out the user's broader taste DNA or seed game affinities.
- **Backlog & discovery Decoupling**: We decoupled "Owned" status from "Discovery" eligibility. Previously, any game in a user's library was excluded from discovery results to prevent redundancy. Build 46 allows games with zero playtime (Backlog) to appear in discovery lists, enabling users to find hidden gems they already own but haven't yet played.
## 18. High-Fidelity Thematic Search (Build 48)

To capture "genre-defying" atmospheric and thematic relationships, we added a third thematic layer using [**BERTopic**](https://maartengr.github.io/BERTopic/). 

- **Topic Discovery**: We trained a topic model on the combined descriptions and reviews of the top 150,000 games on Steam. The model discovered **250 distinct topics** representing specific gameplay loops (e.g., "Deckbuilding + Rogue-lite"), settings ("Underwater + Submarine"), and technical niches ("VR + Physics").
- **Keywords**: Each topic is defined by its top 10 keywords, which are displayed in the UI to provide human-readable "explainability" for the thematic match.
- **JSD-Based Similarity**: Instead of cosine similarity, we measure the distance between games in the topic space using [**Jensen-Shannon Divergence (JSD)**](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence). This is a specialized information-theoretic distance for probability distributions that balances the importance of both primary and secondary themes.
- **Topic Match Slider**: This component provides a high-fidelity "Atmosphere" signal that complements the categorical "Tags" and narrative "Semantic Vibe" sliders.

## 19. Precision DNA Solver (Build 49)

To handle the complexity of solving across three thematic layers (Tags, Semantics, Topics) without overfitting, Build 49 introduced a **Zero-Order Relevance Filter**.

- **Relevance Filtering (p <= N-7)**: Before the LASSO regression runs, the solver calculates the absolute Pearson correlation between every feature and the user's ratings. Only the top $P$ features (where $P$ is constrained by the number of ratings $N$) are allowed into the model. This ensures the solver always has enough "degrees of freedom" to find a stable solution, significantly increasing the training $R^2$ for users with smaller libraries.
- **Population-Correct Scaling**: To ensure that Tags, Semantics, and Topics compete fairly in the sparse model, we normalize each modality based on its **Population Variance**. This prevents "Topics" (which naturally have low variance) from being ignored by the L1 penalty, ensuring your DNA captures your true atmospheric preferences.

## 20. Archetypal Generalization & Identity Isolation (Build 50)

Build 50 introduced a significant refactoring of the "Taste DNA" solver and recommendation kernel to ensure that learned models generalize beyond a user's library and maintain strict mechanical identity.

- **Archetypal Ridge Regression**: To prevent "overfitting" where the model $R^2$ exceeded 0.7 (effectively memorizing the user's library but failing on the broader Steam population), we refactored the solver to use **45 Fixed Structural Archetypes**. These features include 38 Mechanical Identity Group (MIG) memberships, 1 Quality Probit, and 6 Metadata Z-scores. By reducing the feature space from thousands of tags to 45 archetypes, the solver identifies the **Structural DNA** that defines a user's taste, resulting in a more robust and generalizable $R^2 \approx 0.28$.

- **Differentiable Sigmoid-Based Kernel**: The Jackalope kernel transitioned from hard Boolean vetoes to a **Differentiable Soft-Gate** framework. Using a sigmoid transition function $f(x) = \frac{1}{1 + e^{-x}}$, the kernel now calculates "Conflict Potentials" for mechanical and thematic mismatches. This allows for:
    - **Symmetric Anchor Enforcement**: Strict perspective (First-Person, Isometric) and interaction models (CRPG, RPG) are enforced symmetrically. If a candidate game possesses a hard anchor that the seed lacks, it is strictly penalized (0.001x), ensuring structural consistency in results.
    - **Mood Clash Multipliers**: Serious/Emotional seeds now apply a differentiable penalty (up to 0.6x) to "Lighthearted" candidates (Funny, Comedy, Cartoon), preventing jarring tone shifts in recommendations.
    - **Thematic Rescues**: MIG clashes (e.g., matching a Narrative game to a Walking Sim) can be "rescued" if the thematic (Semantic/Topic) similarity is sufficiently high, allowing for creative discovery without breaking mechanical identity.

- **MIG Refinement**: The 38 Mechanical Identity Groups (MIGs) were refined to require **core verbs** (e.g., `SHOOTER` requires `FPS` or `Shooter`). Broad tags like `Exploration` and `Surreal` were removed from core mechanical definitions and moved to the thematic "Vibe" layer, eliminating identity leaks that previously caused false-positive matches (like matching *Disco Elysium* to a first-person puzzle game).

