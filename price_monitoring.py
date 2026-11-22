
import pandas as pd
import joblib

# Load datasets
original_df = pd.read_csv("dataset/pricing_data.csv")
encoded_df = pd.read_csv("dataset/pricing_data_encoded.csv")

# Check that rows match
if len(original_df) != len(encoded_df):
    raise ValueError(
        "Original and encoded datasets have different row counts."
    )

# Load trained model
model = joblib.load("pricepulse_model.pkl")

# Select model input features
X = encoded_df.drop("recommended_price", axis=1)

# Predict recommended prices
predictions = model.predict(X)

# Create monitoring results
results = original_df.copy()
results["recommended_price"] = predictions

# Calculate price change
results["price_change"] = (
    results["recommended_price"] - results["current_price"]
)

results["price_change_percent"] = (
    results["price_change"] / results["current_price"]
) * 100

# Generate price alerts
results["price_alert"] = results["price_change_percent"].apply(
    lambda change:
        "PRICE INCREASE ALERT" if change >= 5
        else "PRICE DECREASE ALERT" if change <= -5
        else "NORMAL PRICE"
)

# Estimate revenue before and after price changes
results["current_revenue"] = (
    results["current_price"] * results["sales_quantity"]
)

results["estimated_revenue"] = (
    results["recommended_price"] * results["sales_quantity"]
)

# Calculate total revenue
current_revenue = results["current_revenue"].sum()
estimated_revenue = results["estimated_revenue"].sum()

revenue_difference = estimated_revenue - current_revenue

# Display results
print("PRICE CHANGE ALERTS")
print("-------------------")

alerts = results[results["price_alert"] != "NORMAL PRICE"]

print("Total Products:", len(results))
print("Products with Price Alerts:", len(alerts))

print("\nPRICE ALERT DETAILS")

print(
    alerts[
        [
            "product_id",
            "current_price",
            "recommended_price",
            "price_alert"
        ]
    ].head(10).to_string(index=False)
)

print("\nREVENUE MONITORING")
print("------------------")

print("Current Estimated Revenue: ₹", round(current_revenue, 2))
print("New Estimated Revenue: ₹", round(estimated_revenue, 2))
print("Estimated Revenue Difference: ₹", round(revenue_difference, 2))

# Save results
results.to_csv("price_monitoring_results.csv", index=False)

print("\nMonitoring results saved successfully!")
