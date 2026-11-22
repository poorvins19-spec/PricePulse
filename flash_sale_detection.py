
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/pricing_data.csv")

# Calculate average sales quantity
average_sales = df["sales_quantity"].mean()

# Calculate standard deviation
sales_std = df["sales_quantity"].std()

# Define flash sale threshold
threshold = average_sales + (2 * sales_std)

# Detect flash sales
df["flash_sale"] = df["sales_quantity"] > threshold

# Display results
flash_sales = df[df["flash_sale"] == True]

print("FLASH SALE DETECTION")
print("--------------------")

print("Average Sales:", round(average_sales, 2))
print("Flash Sale Threshold:", round(threshold, 2))

print("\nFlash Sale Transactions Detected:", len(flash_sales))

if len(flash_sales) > 0:
    print("\nHIGH DEMAND ALERT!")
    print(flash_sales[
        ["product_id", "product_category", "sales_quantity", "current_price"]
    ].head(10).to_string(index=False))
else:
    print("\nNo flash sale detected.")

# Save results
df.to_csv("flash_sale_results.csv", index=False)

print("\nFlash sale results saved successfully!")
