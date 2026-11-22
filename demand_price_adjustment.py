
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/pricing_data.csv")

# Calculate demand levels
average_demand = df["demand"].mean()

def adjust_price(row):
    if row["demand"] > average_demand * 1.2:
        return row["current_price"] * 1.10
    elif row["demand"] < average_demand * 0.8:
        return row["current_price"] * 0.95
    else:
        return row["current_price"]

# Apply demand-based price adjustment
df["demand_adjusted_price"] = df.apply(adjust_price, axis=1)

# Display results
print("PRICEPULSE - DEMAND-BASED PRICE ADJUSTMENT")
print("------------------------------------------")
print(df[[
    "product_id",
    "demand",
    "current_price",
    "demand_adjusted_price"
]].head(10))

# Save results
df.to_csv("demand_price_adjustment_results.csv", index=False)

print("\nResults saved successfully!")
