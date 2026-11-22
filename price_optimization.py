
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

# Load existing dataset
df = pd.read_csv("dataset/pricing_data.csv")

# Estimate price elasticity from the dataset
valid = df[
    (df["current_price"] > 0) &
    (df["sales_quantity"] > 0)
]

elasticity = np.polyfit(
    np.log(valid["current_price"]),
    np.log(valid["sales_quantity"]),
    1
)[0]

print("PRICEPULSE - NONLINEAR PRICE OPTIMIZATION")
print("-----------------------------------------")
print("Estimated Price Elasticity:", round(elasticity, 4))

results = []

for _, row in df.iterrows():

    current_price = row["current_price"]
    base_demand = row["sales_quantity"]
    inventory = row["inventory"]

    # Estimate demand at a proposed price
    def calculate_revenue(price):

        estimated_demand = base_demand * (
            price / current_price
        ) ** elasticity

        units_sold = min(
            max(0, estimated_demand),
            inventory
        )

        return price * units_sold

    # Optimize price within 80%-120% of current price
    result = minimize_scalar(
        lambda price: -calculate_revenue(price),
        bounds=(current_price * 0.8, current_price * 1.2),
        method="bounded"
    )

    optimal_price = result.x

    results.append({
        "product_id": row["product_id"],
        "current_price": current_price,
        "optimal_price": round(optimal_price, 2),
        "optimized_revenue": round(
            calculate_revenue(optimal_price), 2
        )
    })

# Save optimized prices
output = pd.DataFrame(results)
output.to_csv("optimized_prices.csv", index=False)

print("\nOptimization completed!")
print(output.head(10))
print("\nSaved: optimized_prices.csv")
