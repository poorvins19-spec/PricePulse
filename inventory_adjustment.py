
import pandas as pd

# Load competitor monitoring result
df = pd.read_csv("competitor_price_result.csv")

# Load original dataset to get inventory
original_df = pd.read_csv("dataset/pricing_data.csv")

# Select the same sample product
product = original_df.iloc[0]

# Get current values
suggested_price = df["Final_Suggested_Price"].iloc[0]
inventory = product["inventory"]
demand = product["demand"]

# Inventory-based adjustment
if inventory < 50 and demand > 100:
    adjustment = 1.10
    status = "Low inventory - price increased by 10%"

elif inventory > 150:
    adjustment = 0.95
    status = "High inventory - price reduced by 5%"

else:
    adjustment = 1.00
    status = "Normal inventory - no price adjustment"

# Calculate final price
final_price = suggested_price * adjustment

print("INVENTORY-BASED PRICE ADJUSTMENT")
print("--------------------------------")
print("Available Inventory:", inventory)
print("Product Demand:", demand)
print("Previous Suggested Price: ₹", round(suggested_price, 2))
print("Inventory Status:", status)
print("Final Dynamic Price: ₹", round(final_price, 2))

# Save result
result = pd.DataFrame([{
    "Inventory": inventory,
    "Demand": demand,
    "Previous_Suggested_Price": round(suggested_price, 2),
    "Inventory_Status": status,
    "Final_Dynamic_Price": round(final_price, 2)
}])

result.to_csv("inventory_adjustment_result.csv", index=False)

print("\nInventory adjustment result saved successfully!")
