# Methodology

## 1. Data Collection & Preprocessing
The initial dataset consists of Steam games as of March 2025 downloaded from [Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset). Each month Steam is scraped for new content/scores/reviews/descriptions/comments etc. Data is collected directly from the [**Steam Storefront**](https://store.steampowered.com/) and the official [**Steam API**](https://developer.valvesoftware.com/wiki/Steam_Web_API) to ensure high fidelity for user tags and review counts. Review counts are prioritized from **English** language sources where available, falling back to global counts to ensure relevance for English-speaking users while maintaining coverage for international titles.

## 2. Semantic Embeddings
To enable natural language search, we use the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) SentenceTransformer model. To improve precision, we utilize a **Dual Semantic Vector** system.

- **Structural Vector:** Encodes the categorical properties of a game, such as its **Genres** and **Tags**. This captures the "what it is" aspect of the game.

- **Descriptive Vector:** Encodes the narrative and "vibe" of a game using its **Short Description** and **User Reviews**. This captures the "how it feels" and "how it's talked about" aspects of the game.

- **Process:** Each piece of text is converted into a high-dimensional vector that represents its meaning. If the text is a single word, the model encodes the expected context of that word. Groups of words have their vectors pooled, allowing long text to be represented by a single vector.  

- **Usage:** The similarity between these vectors is measured using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), which gives a numerical value indicating how alike their meanings are. This allows the model to recognize synonyms like "peaceful" and "calm" as closely related concepts, even if the words don't match exactly. By calculating the angle between vectors, we can compare the user prompt with both the description and tags of games in the database. When using "Seed" games, the system performs specialized matching: structural seed vectors are compared against other games' structural vectors, and descriptive against descriptive. This prevents "meaning leak"—for example, it ensures that a seed game's narrative description doesn't incorrectly match an unrelated game just because they share a common tag word.

- **Whitening:** To improve semantic precision, we apply [Zero-phase Component Analysis (ZCA) Whitening](https://martin-thoma.com/zca-whitening/) to the semantic vectors. Raw transformer embeddings on specialized sets of text (e.g., video game descriptions) often have high correlation between dimensions and a significant non-zero mean, which can bias searches toward common semantic themes. We first subtract the global mean of the embeddings to center the distribution, then apply ZCA on the mean-centered data to decorrelate dimensions. This ensures that the similarity metric focuses on unique game characteristics and improves the specificity of natural language queries. The stored mean and transformation matrix are both applied to user prompts at query time to maintain alignment in the decorrelated space. 

## 3. Tag Embeddings
Steam tags are user-contributed and often noisy. We apply a pipeline to transform these raw counts into robust "tag vectors" that capture the stylistic profile of each game.

- **Iterative EM Imputation:** To handle unobserved tags (Steam limits API responses to the top 20 tags), we use an Iterative Expectation-Maximization (EM) algorithm. The E-step imputes counts for unobserved tags using the current covariance matrix (conditional expectation), capped by the count of the 20th tag. The M-step recalculates the global mean and covariance from these augmented profiles. This is repeated until the model converges to a stable estimate of the missing data. The result is a best guess for the distribution of censored tags that fall outside of the top 20. 

- **Stochastic Path Optimization:** To determine the optimal regularization constant $K$, we employ a simulation study. We select "reliable" games (with >1,000 votes) and simulate smaller versions of them by drawing samples from the data associated with one game at a time; the number of samples per game is a random variable taken from the sizes from the actual dataset distribution. We then solve for the $K$ that minimizes the Sum of Squared Errors (SSE) between the regularized synthetic profiles and the true reliable profiles.

- **Bayesian Regularization:** We apply Bayesian smoothing using the optimized $K$ to shrink low-information games toward the global prior.

- **Swappable Transformations & Bayesian Regularization:** We support three transformation options to stabilize the variance of tag counts: the [Anscombe Transform](https://en.wikipedia.org/wiki/Anscombe_transform) ($2\sqrt{x + 3/8}$), [Centered Log-Ratio (CLR)](https://en.wikipedia.org/wiki/Compositional_data#Center_log_ratio_transform), or an Identity transform. Currently we are evaluating whether any of these outperforms the others for quality of matches.

- **Process:** We first apply Bayesian regularization directly to the raw tag counts: $\text{Profile} = (C + K \cdot G) / (N + K)$, where $C$ is the count vector, $G$ is the global prior, and $N$ is the total votes. This ensures that the regularization happens on the "natural" scale of the data. We then apply the chosen transformation to both the regularized profile and the global prior. The final embedding is the difference between these two in the transformed space: $V = \text{Transform}(\text{Profile}) - \text{Transform}(G)$. This ensures that the "regularizing point" (a game with zero observed tags) resides at the origin ($0$) of the embedding space, making it a neutral reference for similarity calculations.

![Tag Correlation Carpet Plot](assets/tag_correlation_carpet_plot.png)

- **Regularized Similarity:** We use a **Regularized Cosine Similarity** for tag vectors: $\text{Sim}(A,B) = \frac{A \cdot B}{\|A\|\|B\| + \lambda}$. This effectively penalizes similarity scores for games with low-information (short) tag vectors, ensuring that recommendations are based on strong, confident tag matches. The parameter $\lambda$ is determined by fitting a [Chi-distribution](https://en.wikipedia.org/wiki/Chi-distribution) to the lengths (norms) of "low-tag" vectors (norms between 0 and 5). We set $\lambda$ to the 95th percentile of this fitted distribution, which represents the "noise floor" of the embedding space. This ensures that a vector's length must be statistically significant before it can achieve high similarity scores.

- **Whitening & Dimensionality Reduction:** To ensure stability and remove numerical noise, tag vectors undergo **Truncated PCA-ZCA Whitening**. The data is projected onto its top principal components, which capture 95% of the global variance while eliminating singular dimensions (such as the linear constraint imposed by the CLR transform). This process decorrelates the tags and prevents "ghost" similarities between unrelated games.

## 4. Personalization: Taste DNA
To provide bit-perfect recommendations tailored to individual history, we use a supervised regression pipeline to build a "Taste DNA" profile.

- **Soft-Labeling:** We estimate a user's potential rating for every game in their library using a combination of their explicit Steam reviews and their relative playtime. We apply a **Log-Normal Kernel** to the global playtime distribution of each game to predict the probability of a positive review at the user's specific playtime ($p+(t)$). This probability is blended with the global quality score to produce a predicted rating on a 0-10 scale.

- **LASSO Regression:** We solve for the user's "Taste DNA" using [**LASSO Regression**](https://en.wikipedia.org/wiki/Lasso_(statistics)). This technique is chosen for its ability to perform automatic feature selection; by applying an L1 penalty, the model identifies the specific subset of tags and metadata that truly drive a user's preferences, setting irrelevant features to exactly zero.

- **Adaptive Dimensionality:** To prevent overfitting while maintaining high fidelity, we use an **Adaptive Dimensionality** scheme for the tag space. The number of principal components $K$ used during solving is dynamically set based on the user's library size $N$, typically following $K = \text{clip}(N-6, 1, 243)$. This ensures that users with small libraries receive stable, broad profiles, while power users with thousands of games receive highly granular, high-dimensional taste vectors.

- **Unified Pathway:** Both the **Taste DNA Solver** and the **Discovery Engine** utilize the exact same mathematical code path (`calculate_linear_scores`). This ensures 100% ranking parity between the games used to train the profile and the recommendations surfaced by the engine.

## 5. Quality Scoring
Instead of raw review percentages, we use a Bayesian score that smooths out noisy reviews. This allows the system to distinguish between a game with 10 positive reviews out of 10 and a masterpiece with 98,000 positive reviews out of 100,000. 

The model we use is based around the idea that there is a latent quality score $Q$ for each game, a normally distributed range of possible experiences that users might have with that game, and a threshold where, if a user has an experience greater than the threshold, they will give a positive review. Under some minimal assumptions, this gives the relation $\frac{p}{p + n} = \Phi(Q)$, where $p$ and $n$ are the number of positive and negative reviews left for the game, respectively, and $\Phi$ is the [cumulative distribution function of a standard normal variable](https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function). However, we do not necessarily trust the overall review score, especially when there are only a few reviews; this is due to [the cold start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)). So, we introduce a variable $s$ that "shrinks" the percentage of positive reviews toward a group mean, where $s$ has a larger effect for games with fewer reviews. The variable $s$ can be thought of as a set of initial reviews shared by all games that reduces the variability of estimates of $Q$ when there are few reviews. The resulting regularization scheme yields the following formula: $\frac{p + s \cdot a}{p + n + s} = \Phi(Q)$; solving for $Q$ gives $Q = \Phi^{-1}\left(\frac{p + s \cdot a}{p + n + s}\right)$.  

- **Components:** $p$ and $n$ are positive and negative reviews, $a$ is the global positive review rate derived from the dataset, and $s$ is a regularization constant.

- **Tunable Discovery:** The "Discovery" slider allows users to adjust $s$ in real-time. A high discovery setting ("Wild Cards") uses a low $s$, allowing "hidden gems" with few reviews to climb the rankings, while a low discovery setting ("Known Quantities") uses a high $s$ to ensure only established, highly-reviewed titles reach the top. The application uses a high-resolution precalculated grid (201 steps) to provide smooth transitions between these settings.

- **Probit Function:** $\Phi^{-1}$ is the [probit function](https://en.wikipedia.org/wiki/Probit), which converts probabilities into a linear scale (z-scores), making the differences between "great" and "perfect", or "bad" and "horrendous", more meaningful.

- **Score Normalization:** To maintain a consistent spread of scores in recommended games across different Discovery settings, we apply a linear transformation to the generated z-scores. This ensures that the mean scores of the top 500 and bottom 500 games remain aligned with a baseline distribution (s = 2000), preventing the score distribution from collapsing or expanding excessively at extreme parameter values.

- **Computational Efficiency:** To minimize memory usage on resource-constrained environments, the recommendation engine utilizes [Reduced Precision Arithmetic](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) (FP16) for large-scale vector operations and [Memory-Mapped I/O](https://en.wikipedia.org/wiki/Memory-mapped_file) for static data artifacts. This allows the system to perform high-dimensional similarity searches efficiently.

## 6. Playtime Regularization
To effectively rank games by length, we analyze the distribution of playtimes that are available as part of user reviews. We assume that negative reviews will bias the estimate of game length downward due to quitting the game before finishing it, so we discard those values. Median playtime estimates can be highly variable for games with few reviews, and to address this, we apply a Bayesian shrinkage method similar to our tag vector approach.

- **Stochastic Path Analysis:** We determine an optimal regularization constant $C$ by creating synthetic "small" games using reviews from "reliable" games (those with $\ge 80$ positive reviews). We solve for the $C$ that minimizes the error between the regularized estimate of the small game and the true median of the original reliable game.

- **Metric:** We regularize the logarithm of the median playtime: $\log(1 + \text{median})$.

- **Formula:** $\text{Regularized Log-Median} = \frac{n \cdot \text{Sample Log-Median} + C \cdot \text{Global Mean Log-Median}}{n + C}$, where $n$ is the number of positive reviews. This ensures that games with very few reviews are strongly pulled toward the global average, while games with substantial feedback retain their specific playtime identity.

- **Display:** The exponential of this regularized value is displayed as the estimation of length. 

## 7. Difficulty Prediction
Steam games do not natively have a "Difficulty" rating. To solve this, we built a predictive model using external data fitted to Steam game tags. 

- **Source Data:** We use an external dataset of ~3,200 games with explicit difficulty ratings that are in the Steam ecosystem.

- **Model:** We train a linear regression model to predict this difficulty score using Steam tags as features.

    - **Transformation:** Tag proportions and difficulty scores are transformed using [**Rank-Based Inverse Normal Transformation (Rank-INT)**](https://github.com/alexjamesing/RankBasedInverseNormal) to handle the sparse and non-normal distribution of tag data.

    - **Feature Selection:** We used [**Forward/Backward Iterative Feature Selection**](https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/) with [**Bayesian Information Criterion (BIC)**](https://en.wikipedia.org/wiki/Bayesian_information_criterion) to identify significantly predictive tags in the external data. Some of the significant predictors are obvious tags like ["Difficult"](https://store.steampowered.com/tags/en/Difficult), ["Precision Platformer"](https://store.steampowered.com/tags/en/Precision%20Platformer), and ["Souls-like"](https://store.steampowered.com/tags/en/Souls-like). There are also less obvious ones like ["Visual Novel"](https://store.steampowered.com/category/visual_novel) and ["Short"](https://store.steampowered.com/tags/en/Short), which predict easy games.  

- **Estimation** We apply this model to all steam games. The result is a difficulty prediction value from 0 to 10, where 0 indicates a very easy game and 10 indicates a very difficult one. The mean difficulty estimate for all Steam games is around 5.5, and our prediction for the most difficult game on Steam is [Tametsi](https://store.steampowered.com/app/709920/Tametsi/). Play it if you hate yourself.

- **Z-Scoring:** For the final difficulty rating, the predicted difficulty estimate is converted to a z-score relative to the distribution of all games that have tags. This allows users to weight games by their estimated challenge level on the same scale as other features, such as popularity or length.

- **Display:** The predicted 0-10 difficulty score is displayed directly on the game cards, providing a quick estimate of the expected challenge.

## 8. Content Sensitivity (NSFW Blur)
To maintain a safe and professional discovery experience, Steam Jackalope employs an **"NSFW Blur" Architecture**. 

- **Flagging:** Games are automatically flagged as `is_nsfw` based on a curated list of sensitivity tags (e.g., "Hentai", "Sexual Content", "Nudity"). 

- **UI Interaction:** Instead of hard-filtering results, which can bias the algorithm's understanding of "vibe," the frontend uses a global **Blur Toggle**. When active, any game flagged as NSFW has its header image blurred using a CSS backdrop filter. This maintains the integrity of the recommendation list while giving users control over their visual environment.

## 9. The Jackalope

The final recommendation list is generated as a hybrid, blending normalized components:

1.  **Semantic Match:** Similarity between the natural language prompt and game text. This component is Z-scored to maintain a consistent scale relative to other features.

2.  **Tag Match:** Similarity between the latent tag vectors. In Build 16, this was unified with the Linear Scorer model. Every game $V$ used as a seed provides a set of regression coefficients $\beta_{seed} = V / (\|V\| + \lambda)$. The contribution is calculated as the penalized dot product $Score = (U \cdot \beta_{seed} / (\|U\| + \lambda)) \cdot 11.283$. This component is no longer Z-scored, as it operates on an absolute 0-10 rating scale calibrated to the user's taste.

3.  **Quality Score:** Preference for loved vs. hated games, smoothed by the Discovery setting.

4.  **Popularity:** Preference for high vs. low player counts.

5.  **Game Age:** Preference for new vs. classic titles.

5.  **Length:** Preference for short vs. long experiences.

6.  **Difficulty Rating:** Preference for Easy vs. Difficult games.

Users can tune these weights in the UI to prioritize long, easy, niche hidden gems or popular classics that are similar to a game of choice. 

**Data Reliability & Quality Adjustments:**

- **Review Count Repair:** To handle stale metadata, the system automatically repairs global review counts using raw individual reviews if our scrape finds more data than the storefront summary.

- **Age Stability:** Future release dates are clamped to the current date, and unknown dates are assigned a neutral z-score (0) to maintain distribution stability.

- **Global Clamping:** To prevent extreme outliers from disproportionately influencing the final score, all component z-scores are clamped between -8.0 and 8.0.

## 8. Filtering
The system allows real-time filtering for **Genres**, VR-Only titles, English language support, NSFW content (Adult Only banner), Software/Utilities (Breadcrumb detection), and Unreleased games to ensure the results are relevant to the user. The Genre filter supports multi-selection, ensuring that recommended games match at least one of the selected categories. Downloadable Content (DLC) is excluded from the database to focus recommendations on standalone games.

## 9. Insights & Rankings
The "Lists" page provides a curated view of the Steam library's extremes, including the highest and lowest quality games, the longest and shortest experiences, and predicted difficulty rankings. These lists utilize the same Bayesian models and predictive analytics used in the recommendation engine, providing transparency into how different games are positioned within our statistical model.

The "Lists" page also includes a **Similarity Analysis** tab. This tool identifies popular yet diverse seed games (tag similarity < 0.2) and displays their most similar matches based on both tag and semantic embeddings. This provides a direct look at the engine's core similarity logic, demonstrating how it handles both categorical (tag-based) and stylistic (semantic) matches for well-known titles.

## 10. Playtime-Sentiment Correlation
To understand how player engagement relates to satisfaction, we utilize a **Kernel Smoothing** model to analyze the relationship between playtime and review sentiment (positive vs. negative). This allows us to estimate the probability that a player will enjoy a game based on how long they have played it.

- **Lognormal Kernel:** We use a Gaussian kernel in log-space to measure similarity between playtimes. This accounts for the fact that the difference between 1 hour and 2 hours is more significant than the difference between 100 and 101 hours.

- **Leave-One-Out Prediction:** For each review in a game's history, we estimate its probability of being positive by taking a weighted vote of all *other* reviews for that game. Reviews from players with similar playtimes carry the most weight.

- **Bayesian Regularization:** To handle timeframes with few reviews, we apply Bayesian smoothing, pulling the predicted probability toward the global average positive review rate (0.80).

- **Optimization:** The model's smoothing bandwidth ($\gamma$) and regularization strength ($s$) were globally optimized using a parallelized grid search over the entire dataset of 36,000+ games. We maximized the total log-likelihood (cross-entropy) of the historical review data, ensuring that every review across all games contributes equally to the calibration. 

- **Usage:** This model reveals patterns such as "early quitters" (negative reviews at low playtime) vs. "burnout" (negative reviews at extremely high playtime), providing deeper insights into the player experience beyond a single aggregate score.

## 11. Visual & UI Stability
To provide a polished and safe user experience, the modern frontend implements several automated visual treatments:

- **NSFW Content Blurring:** Games flagged with the "Adult Only" or "Mature" content banners (or those containing specific NSFW keyword patterns) automatically have their banner images blurred in the recommendation grid. Users can click a "Reveal" button on individual cards to temporarily clear the blur.

- **Grid Rendering Stability:** The recommendation grid utilizes persistent React keys based on game AppIDs and standardized CSS layout principles (avoiding flexbox on global containers) to prevent "Single Card" collapse bugs and ensure consistent card sizing across all screen resolutions.

- **Dynamic Tag Visualization:** Each game card dynamically renders its most relevant genres and top user tags, highlighting matches that align with the user's search intent or seed games.

## 12. Personalized Linear Scorer (Build 15)

To ensure perfect alignment between a user's library analysis and their discovery feed, Build 15 introduced a **Unified Linear Scorer** architecture. This replaces the "hybrid approximation" with a direct execution model.

- **Unified Feature Space:** Both the Taste DNA Solver and the Recommendation Engine utilize a standardized global feature space. Metadata (Age, Quality, Popularity, etc.) are represented as **Global Z-scores**, while Steam tags are transformed using **Penalized Normalization** ($v / (\|v\| + \lambda)$) and then scaled by a global constant (**11.283x**) to match the variance of the Z-scored metadata.

- **Portable Beta Weights:** By removing user-specific scaling, the DNA Solver learns **Beta Weights** that are directly portable. These weights represent the absolute importance of each feature (e.g., "how many rating points is one unit of Difficulty worth to this specific user?").

- **Direct Execution:** When a Taste Profile is applied, the Recommendation Engine switches into **Linear Mode**. Instead of using internal biases, it calculates the final Match Score by directly running the user's solved regression model: $\text{Match Score} = \text{Intercept} + \sum (\beta_i \cdot x_i)$. 

- **Predictive Sliders:** In this mode, the UI sliders transition from absolute biases to **Multipliers**. A slider at 0.5 uses the solved DNA weight exactly (100%), while 0.0 ignores the preference and 1.0 doubles its impact. This allows users to fine-tune their DNA in real-time.

- **Predicted Ratings:** Because the math is unified, the "Match Score" displayed on game cards becomes a calibrated **Predicted 0-10 Rating** for that game, ensuring that the "Games You'll Love" list in the analyzer and the Recommendation results are always mathematically identical.

## 13. Robust Personalization (Build 36)

As the personalization engine matured, we introduced several features to ensure that the "Taste DNA" remains both statistically sound and human-readable.

- **Adaptive DNA Dimensionality:** To prevent the model from "overfitting" or memorizing small libraries, the solver dynamically scales the complexity of the tag space based on the number of ratings provided by the user. We use a smooth linear relationship: $K = \text{clamp}(40 + 0.7 \times N_{\text{ratings}}, \text{min}=40, \text{max}=243)$. This ensures that new users with few ratings are modeled using only broad, high-certainty genre components, while power users with hundreds of ratings gain access to high-fidelity, niche stylistic details.

- **Support-Based Tag Filtering:** High-dimensional models can sometimes create "phantom" associations (statistical aliasing) where a tag you haven't actually played shows up in your DNA because it is correlated with something you like. To solve this, we implemented a **Sanity Check** layer. A tag is only eligible to be displayed in the "Love/Hate" lists if it has direct evidence in your library (i.e., it appears in at least one game you have rated). This ensures that the DNA view is always grounded in your actual play history.

- **Scoring Synchronization:** To ensure 100% parity between the Solver's preview and the Recommender tool, we synchronized all implementation details, including bit-perfect tag normalization (using pre-calculated norms), Z-score clamping at $\pm 8.0$, and lexicographical tie-breaking (Score DESC, Name ASC).

## 14. Data Hygiene & External Integration (Build 13)

To ensure the recommender remains a useful portal to the Steam ecosystem, Build 13 introduced a verified external link layer:

1. **Steam Store Mapping**: A validation process (`tools/validate_steam_links.py`) maps Steam Store patterns (`/tags/en/`, `/genre/`, `/category/`, and feature-specific search IDs) to verify valid landing pages for every term in our database.
2. **Dead Tag Filtering**: Terms that do not map to a valid Steam Store page are flagged as "dead." The backend filters these from the game metadata during the loading phase. This removes junk data and legacy metadata from the discovery engine.
3. **Interactive Taxonomy**: Verified terms are served as a linkable mapping, enabling the UI to render tags and genres as clickable pathways directly to the Steam community hubs.
