
import pandas as pd

# Load sales dataset
df = pd.read_csv("dataset/pricing_data.csv")

# Create simulated timestamps (one event per second)
df["event_time"] = pd.date_range(
    start="2026-01-01 09:00:00",
    periods=len(df),
    freq="s"
)

# Set timestamp as index
df = df.set_index("event_time")

# Aggregate sales every minute
minute_demand = df["sales_quantity"].resample("1min").sum()

# Calculate 5-minute sliding demand windows
sliding_demand = minute_demand.rolling(
    window=5,
    min_periods=5
).sum()

# Create results
results = pd.DataFrame({
    "minute_demand": minute_demand,
    "five_minute_demand": sliding_demand
})

# Remove incomplete initial windows
results = results.dropna()

# Calculate high-demand threshold
threshold = (
    results["five_minute_demand"].mean()
    + 2 * results["five_minute_demand"].std()
)

# Detect demand spikes
results["demand_alert"] = results["five_minute_demand"].apply(
    lambda value: "HIGH DEMAND ALERT"
    if value > threshold else "NORMAL DEMAND"
)

# Display results
print("DEMAND SLIDING WINDOW ANALYSIS")
print("--------------------------------")
print("Window Duration: 5 minutes")
print("Slide Duration: 1 minute")
print("Demand Threshold:", round(threshold, 2))

print("\nSLIDING WINDOW RESULTS")
print(results.tail(10).to_string())

print(
    "\nHigh Demand Windows:",
    (results["demand_alert"] == "HIGH DEMAND ALERT").sum()
)

# Save results
results.to_csv("demand_sliding_window_results.csv")

print("\nSliding window results saved successfully!")
