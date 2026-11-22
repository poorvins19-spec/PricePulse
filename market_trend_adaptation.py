
import pandas as pd
import joblib

# Load existing dataset and trained model
data = pd.read_csv("dataset/pricing_data_encoded.csv")
model = joblib.load("pricepulse_model.pkl")

# Use exactly the features used during model training
features = list(model.feature_names_in_)
X = data[features]

# Predict the base price using the ML model
data["ML_Predicted_Price"] = model.predict(X)

# Calculate competitor market trend
data["Market_Trend"] = (
    (data["competitor_price"] - data["current_price"])
    / data["current_price"]
)

# Limit the trend adjustment to avoid extreme price changes
data["Market_Trend"] = data["Market_Trend"].clip(-0.10, 0.10)

# Adjust the ML price using market trend
data["Trend_Adapted_Price"] = (
    data["ML_Predicted_Price"] * (1 + 0.20 * data["Market_Trend"])
)

# Save results
data[
    [
        "current_price",
        "competitor_price",
        "ML_Predicted_Price",
        "Market_Trend",
        "Trend_Adapted_Price"
    ]
].to_csv("market_trend_results.csv", index=False)

print("MARKET TREND ADAPTATION")
print("-----------------------")
print(data[
    [
        "current_price",
        "competitor_price",
        "ML_Predicted_Price",
        "Trend_Adapted_Price"
    ]
].head(5))

print("\nResults saved to market_trend_results.csv")
