
import pandas as pd
import joblib

# Load dataset and trained model
df = pd.read_csv("dataset/pricing_data_encoded.csv")
model = joblib.load("pricepulse_model.pkl")

# Select a sample product
product = df.iloc[[0]]

# Separate input features
X = product.drop("recommended_price", axis=1)

# Predict price using Random Forest
ml_price = model.predict(X)[0]

# Get competitor price
competitor_price = product["competitor_price"].iloc[0]

# Competitor price blending
final_price = (0.70 * ml_price) + (0.30 * competitor_price)

print("COMPETITOR PRICE MONITORING")
print("---------------------------")

print("ML Recommended Price: ₹", round(ml_price, 2))
print("Competitor Price: ₹", round(competitor_price, 2))
print("Final Suggested Price: ₹", round(final_price, 2))

if final_price > competitor_price:
    print("\nOur suggested price is higher than competitor price.")
elif final_price < competitor_price:
    print("\nOur suggested price is lower than competitor price.")
else:
    print("\nOur suggested price matches competitor price.")

# Save monitoring result
result = pd.DataFrame([{
    "ML_Price": round(ml_price, 2),
    "Competitor_Price": round(competitor_price, 2),
    "Final_Suggested_Price": round(final_price, 2)
}])

result.to_csv("competitor_price_result.csv", index=False)

print("\nCompetitor monitoring result saved successfully!")
