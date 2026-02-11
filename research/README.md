# Research Directory

This directory contains experimental scripts, data analysis notebooks, and exploratory code used to develop and validate the statistical models behind SteamJackalope.

## Overview

Research scripts are used for:

- Testing hypotheses about game features and user behavior
- Developing and tuning predictive models (e.g., difficulty prediction)
- Analyzing distributions and correlations in the data
- Running simulation studies to optimize regularization constants
- Generating diagnostic plots and visualizations

## Organization

Scripts in this directory are typically:

- **One-off analyses**: Exploratory work that informed design decisions
- **Model development**: Training and evaluating predictive models
- **Validation studies**: Checking assumptions and stability of transformations
- **Simulation tools**: Studying parameter sensitivity and impact

## Notable Scripts

### Difficulty Prediction

- `train_difficulty_model.py` - Main script for training the difficulty prediction model
- `aic_difficulty.py`, `bic_difficulty_parallel.py` - Information criterion-based feature selection
- `lasso_difficulty.py`, `ridge_difficulty.py` - Regularized regression approaches
- `supervised_pca_difficulty.py` - Dimension reduction with outcome supervision
- `match_gamefaqs_to_steam.py`, `parse_gamefaqs.py` - External data integration

### Tag Analysis

- `analyze_tags.py` - Tag distribution and coverage analysis
- `count_tags.py` - Tag frequency statistics
- `sanity_check_tags.py` - Data quality checks
- `factor_analyze_tags.py` - Latent factor extraction from tag space

### Distribution Analysis

- `analyze_vector_distributions.py` - Verify normal distribution assumptions
- `analyze_difficulty_distribution.py` - Examine difficulty score skewness
- `analyze_embeddings.py` - Semantic vector properties
- `analyze_difficulty_correlations.py` - Correlate difficulty with tags

### Simulation Studies

- `simulate_slider_impact.py` - Quantify recommendation sensitivity to slider adjustments
- `solve_tag_regularization.py` - Determine optimal `TAG_VECTOR_K`
- `iterative_bic_refinement.py` - Stepwise model selection

### Visualization

- `generate_diagram.py`, `generate_cosine_diagram.py` - Create figures for documentation
- Various `.png` files (e.g., `difficulty_distribution.png`, `leverage_analysis.png`)

## When to Add New Research Scripts

- When testing a new algorithmic approach before productionizing it
- When performing deep data analysis to understand limitations
- When running large-scale simulations to tune parameters
- When generating diagnostic plots for documentation

**Note:** Once a research script matures into a production pipeline component, it should be moved to `pipeline/` and documented there.

## Running Research Scripts

Most scripts are standalone and can be run directly:

```bash
python research/analyze_difficulty_distribution.py
```

Check individual script headers for specific dependencies and output formats.

## Dependencies

Research scripts often use additional libraries not in main requirements:

- `matplotlib`, `seaborn` for plotting
- `scikit-learn` for advanced modeling
- `statsmodels` for statistical tests
- Jupyter-related packages for notebooks (if any)

These are typically already in `requirements.txt` or can be added.

## Outputs

Research outputs include:

- Diagnostic plots (saved as `.png`)
- Data analysis results (printed to console or saved as `.csv`/`.json`)
- Model parameters (may feed into `pipeline/regularization_constants.json`)
- Insights that inform `methodology.md` updates

## Related

- See `pipeline/` for production implementations of research outputs
- See `methodology.md` for theoretical explanations of the models developed here
- See `orientation.md` for project context and current research findings