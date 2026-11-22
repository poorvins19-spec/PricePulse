
import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("dataset/pricing_data.csv")

# Calculate price and demand averages
average_price = df["current_price"].mean()
average_demand = df["demand"].mean()

# Calculate correlation
correlation = df["current_price"].corr(df["sales_quantity"])

# Estimate elasticity using log-log relationship
valid_data = df[
    (df["current_price"] > 0) &
    (df["sales_quantity"] > 0)
].copy()

valid_data["log_price"] = np.log(valid_data["current_price"])
valid_data["log_sales"] = np.log(valid_data["sales_quantity"])

elasticity = np.polyfit(
    valid_data["log_price"],
    valid_data["log_sales"],
    1
)[0]

print("PRICE ELASTICITY ANALYSIS")
print("-------------------------")

print("Average Product Price:", round(average_price, 2))
print("Average Demand:", round(average_demand, 2))
print("Price and Sales Correlation:", round(correlation, 3))
print("Estimated Price Elasticity:", round(elasticity, 3))

if elasticity < -1:
    print("\nResult: Demand is relatively price-sensitive.")
elif elasticity < 0:
    print("\nResult: Demand is relatively less price-sensitive.")
else:
    print("\nResult: No negative price-demand relationship detected.")

print("\nPrice elasticity analysis completed!")
