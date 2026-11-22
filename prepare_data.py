
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/pricing_data.csv")

print("Original Dataset Shape:", df.shape)

# Remove duplicate and missing records
df = df.drop_duplicates()
df = df.dropna()

# Select input features
features = [
    "product_category",
    "current_price",
    "demand",
    "inventory",
    "competitor_price",
    "season",
    "customer_segment"
]

# Target column
target = "recommended_price"

# Create a new dataset
data = df[features + [target]].copy()

# Label Encoding for categorical columns
categorical_columns = [
    "product_category",
    "season",
    "customer_segment"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder

# Save prepared dataset
data.to_csv("dataset/pricing_data_encoded.csv", index=False)

print("\nFeature Engineering Completed!")
print("\nPrepared Dataset Shape:", data.shape)

print("\nEncoded Dataset:")
print(data.head())

print("\nCategorical Columns Encoded Successfully:")
print(categorical_columns)

joblib.dump(encoders, "label_encoders.pkl")
print("Label encoders saved successfully!")
print("\nPrepared CSV saved successfully!")
