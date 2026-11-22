
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = pd.read_csv("dataset/pricing_data_encoded.csv")

# Separate input and target
X = data.drop("recommended_price", axis=1)
y = data["recommended_price"]

# Use the same testing split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained models
rf_model = joblib.load("pricepulse_model.pkl")
xgb_model = joblib.load("pricepulse_xgboost.pkl")

# Compare predictions
models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

results = []

for name, model in models.items():
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    })

# Save comparison
comparison = pd.DataFrame(results)
comparison.to_csv("model_comparison.csv", index=False)

print("PRICEPULSE MODEL COMPARISON")
print("---------------------------")
print(comparison.to_string(index=False))
print("\nSaved: model_comparison.csv")
