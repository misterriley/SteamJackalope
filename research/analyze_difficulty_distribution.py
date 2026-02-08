import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('research/bic_stepwise_predictions.csv')

# Set up the figure
plt.figure(figsize=(12, 6))

# Plot Actual
plt.subplot(1, 2, 1)
sns.histplot(df['actual'], bins=100, kde=True, color='blue')
plt.title('Distribution of Actual y-values\n(Blended Difficulty Z-Score)')
plt.xlabel('Actual y')
plt.ylabel('Frequency')
plt.yscale('log') # Use log scale because of the heavy tail

# Plot Predicted
plt.subplot(1, 2, 2)
sns.histplot(df['predicted'], bins=100, kde=True, color='red')
plt.title('Distribution of Predicted y-values\n(BIC Stepwise Model)')
plt.xlabel('Predicted y')
plt.ylabel('Frequency')
plt.yscale('log')

plt.tight_layout()
plt.savefig('research/difficulty_distribution.png')
print("Saved distribution plot to research/difficulty_distribution.png")

# Calculate additional stats
print("\n--- Percentiles ---")
percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
actual_p = df['actual'].quantile([p/100 for p in percentiles])
pred_p = df['predicted'].quantile([p/100 for p in percentiles])

comparison = pd.DataFrame({
    'Percentile': percentiles,
    'Actual': actual_p.values,
    'Predicted': pred_p.values
})
print(comparison.to_string(index=False))

# Check zero/baseline concentration
actual_baseline = df['actual'].mode()[0]
pred_baseline = df['predicted'].mode()[0]
print(f"\nBaseline (Mode) Actual: {actual_baseline:.4f}")
print(f"Baseline (Mode) Predicted: {pred_baseline:.4f}")
print(f"Games at baseline actual: {(df['actual'] == actual_baseline).sum()} ({100*(df['actual'] == actual_baseline).mean():.2f}%)")
print(f"Games near baseline predicted (+/- 0.01): {((df['predicted'] - pred_baseline).abs() < 0.01).sum()} ({100*((df['predicted'] - pred_baseline).abs() < 0.01).mean():.2f}%)")
