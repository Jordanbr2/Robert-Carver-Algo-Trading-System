import numpy as np
import pandas as pd

# Correlation matrix
corr = pd.DataFrame({
    "GDX": [1.0, 0.213, 0.140],
    "SPY": [0.213, 1.0, -0.302],
    "TLT": [0.140, -0.302, 1.0]
}, index=["GDX", "SPY", "TLT"])

# Get upper triangle (excluding diagonal)
vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()

# Cap negatives at 0
vals = np.maximum(vals.values, 0)

# Average correlation
avg_corr = np.mean(vals)

# Carver's Table 18 for 3 assets
carver_table = {
    0.0: 1.73,
    0.25: 1.41,
    0.5: 1.22,
    0.75: 1.12,
    1.0: 1.0
}
closest = min(carver_table, key=lambda x: abs(x - avg_corr))
multiplier = carver_table[closest]

print(f"Average Correlation: {avg_corr:.2f}")
print(f"Diversification Multiplier: {multiplier}")
