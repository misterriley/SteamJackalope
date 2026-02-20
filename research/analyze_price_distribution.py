
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import re
import os

# Load metadata
metadata_path = 'data/production/metadata.parquet'
df = pd.read_parquet(metadata_path)

def parse_price_clean(p):
    if pd.isna(p) or p == "": return None
    s = str(p).lower().strip()
    
    # If it contains "demo", "pass", "users only", "preview", it's likely not a price
    if any(word in s for word in ["demo", "pass", "only", "preview", "minutes", "alpha", "beta"]):
        return None
        
    if "free" in s: return 0.0
    
    # Try to find $XX.XX or XX.XX
    # Look for patterns that look like prices: $19.99, 19.99, $5, 15€
    match = re.search(r'(\d+[.,]\d{2})', s) # XX.XX or XX,XX
    if match:
        val = match.group(1).replace(',', '.')
        try: return float(val)
        except: pass
        
    # Just a number? Only if it's small (e.g. < $200) and doesn't look like an ID (long)
    match = re.search(r'^\s*\$?(\d+)\s*$', s)
    if match:
        val = float(match.group(1))
        if val < 200:
            return val
            
    return None

df['price_numeric'] = df['price'].apply(parse_price_clean)

# Count games without prices
no_price_count = df['price_numeric'].isna().sum()
total_games = len(df)
print(f"Total Games: {total_games}")
print(f"Games without a listed price: {no_price_count} ({no_price_count/total_games:.1%})")

# Filter for games WITH prices for the distribution
priced_games = df.dropna(subset=['price_numeric']).copy()
free_count = (priced_games['price_numeric'] == 0).sum()
print(f"Games listed as Free: {free_count}")

# Setup plot
plt.figure(figsize=(12, 6))

# KDE of all priced games (including Free) - Cap at $60 for better granularity
cap = 60
data = priced_games[priced_games['price_numeric'] <= cap]['price_numeric'].values

if len(data) > 1:
    # Use a small bandwidth to capture the "spikes" at $4.99, $9.99 etc.
    kde = gaussian_kde(data, bw_method=0.05) 
    x_range = np.linspace(0, cap, 1000)
    plt.plot(x_range, kde(x_range), color='#6366f1', lw=2, label='Price Density (KDE)')
    plt.fill_between(x_range, kde(x_range), color='#6366f1', alpha=0.2)

# Add markers for common price points
common_prices = [0.99, 4.99, 9.99, 14.99, 19.99, 29.99, 39.99, 49.99, 59.99]
for p in common_prices:
    plt.axvline(p, color='red', linestyle='--', alpha=0.2, lw=1)
    # plt.text(p, plt.gca().get_ylim()[1]*0.9, f"${p}", color='red', fontsize=8, rotation=90, va='top', ha='right')

plt.title('Steam Price Density Distribution (Capped at $60)', fontsize=14, fontweight='bold')
plt.xlabel('Price (USD)', fontsize=12)
plt.ylabel('Relative Frequency (Density)', fontsize=12)
plt.xticks(np.arange(0, cap+1, 5))
plt.grid(alpha=0.1)
plt.legend()

# Save the plot
output_plot = 'research/price_density_distribution.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_plot}")

# Statistical summary
print("\nCleaned Price Statistics (Priced Games Only):")
print(priced_games['price_numeric'].describe())
