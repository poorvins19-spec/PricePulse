
import pandas as pd

# Load the original CSV dataset
data = pd.read_csv("dataset/pricing_data.csv")

print("PRIVACY-FIRST PRICING ENGINE")
print("---------------------------")

# Keep only the product-level information needed for pricing
safe_data = data[
    [
        "product_category",
        "season",
        "current_price",
        "demand",
        "inventory",
        "competitor_price",
        "sales_quantity",
        "recommended_price"
    ]
].copy()

# Aggregate information instead of keeping individual records
aggregated_data = safe_data.groupby(
    ["product_category", "season"]
).agg({
    "current_price": "mean",
    "demand": "mean",
    "inventory": "mean",
    "competitor_price": "mean",
    "sales_quantity": "sum",
    "recommended_price": "mean"
}).reset_index()

# Save privacy-safe data
aggregated_data.to_csv("privacy_safe_pricing_data.csv", index=False)

print("\nIndividual customer information is not used.")
print("Product-level data has been aggregated.")
print("\nPrivacy-safe pricing data:")
print(aggregated_data.head())

print("\nSaved: privacy_safe_pricing_data.csv")
