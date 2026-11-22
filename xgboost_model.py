
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load prepared dataset
df = pd.read_csv("dataset/pricing_data_encoded.csv")

# Separate features and target
X = df.drop("recommended_price", axis=1)
y = df["recommended_price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create XGBoost model
model = XGBRegressor(
    random_state=42,
    objective="reg:squarederror"
)

# Hyperparameter options
parameters = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1]
}

# Find the best settings
print("Starting XGBoost Hyperparameter Tuning...")

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Test predictions
predictions = best_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nBEST PARAMETERS")
print(grid_search.best_params_)

print("\nTUNED XGBOOST RESULTS")
print("---------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save tuned model
joblib.dump(best_model, "pricepulse_xgboost.pkl")

print("\nTuned XGBoost model saved successfully!")
