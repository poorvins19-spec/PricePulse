
import streamlit as st
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PricePulse",
    page_icon="💰",
    layout="wide"
)

st.title("💰 PricePulse")
st.subheader("Adaptive Dynamic Pricing & Revenue Optimization Engine")

# Create dashboard tabs
tab1, tab2 = st.tabs([
    "Price Prediction",
    "Price & Revenue Monitoring"
])

# -----------------------------
# TAB 1: PRICE PREDICTION
# -----------------------------

with tab1:

    st.header("Product Information")

    product_category = st.selectbox(
        "Product Category",
        ["Beauty", "Electronics", "Fashion", "Grocery", "Home", "Sports"]
    )

    current_price = st.number_input(
        "Current Price (₹)", min_value=1.0, value=500.0
    )

    demand = st.number_input(
        "Product Demand", min_value=0, value=100
    )

    inventory = st.number_input(
        "Available Inventory", min_value=0, value=200
    )

    competitor_price = st.number_input(
        "Competitor Price (₹)", min_value=1.0, value=550.0
    )

    season = st.selectbox(
        "Season", ["Festival", "Regular", "Summer", "Winter"]
    )

    customer_segment = st.selectbox(
        "Customer Segment", ["Budget", "Premium", "Regular"]
    )

    if st.button("Predict Recommended Price"):

        input_data = {
            "product_category": product_category,
            "current_price": current_price,
            "demand": demand,
            "inventory": inventory,
            "competitor_price": competitor_price,
            "season": season,
            "customer_segment": customer_segment
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=input_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                if "recommended_price" in result:
                    price = result["recommended_price"]

                    st.success("Price Prediction Completed!")

                    st.metric(
                        "Recommended Product Price",
                        f"₹{price:.2f}"
                    )

                    difference = price - current_price

                    if difference > 0:
                        st.write(f"Suggested increase: ₹{difference:.2f}")
                    elif difference < 0:
                        st.write(f"Suggested decrease: ₹{abs(difference):.2f}")
                    else:
                        st.write("No price change suggested.")

                else:
                    st.error(result.get("error", "Prediction failed."))

            else:
                st.error("API request failed.")

        except requests.exceptions.RequestException:
            st.error("Cannot connect to FastAPI. Start the API server first.")


# -----------------------------
# TAB 2: PRICE MONITORING
# -----------------------------

with tab2:

    st.header("Price Change Alerts & Revenue Monitoring")

    file_path = "price_monitoring_results.csv"

    if os.path.exists(file_path):

        df = pd.read_csv(file_path)

        # Revenue metrics
        current_revenue = df["current_revenue"].sum()
        estimated_revenue = df["estimated_revenue"].sum()
        revenue_difference = estimated_revenue - current_revenue

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Current Estimated Revenue",
            f"₹{current_revenue:,.2f}"
        )

        col2.metric(
            "Recommended Price Revenue",
            f"₹{estimated_revenue:,.2f}"
        )

        col3.metric(
            "Revenue Difference",
            f"₹{revenue_difference:,.2f}"
        )

        # Price alerts
        st.subheader("Price Change Alerts")

        alerts = df[df["price_alert"] != "NORMAL PRICE"]

        st.write("Total Products:", len(df))
        st.write("Products with Price Alerts:", len(alerts))

        st.dataframe(
            alerts[
                [
                    "product_id",
                    "current_price",
                    "recommended_price",
                    "price_change_percent",
                    "price_alert"
                ]
            ],
            use_container_width=True
        )

        # Revenue Comparison Graph
        st.subheader("Revenue Comparison Graph")

        fig, ax = plt.subplots(figsize=(9, 5))

        labels = ["Current Revenue", "Recommended Revenue"]
        values = [current_revenue, estimated_revenue]

        bars = ax.bar(labels, values)

        ax.set_title("PricePulse Revenue Comparison")
        ax.set_xlabel("Revenue Type")
        ax.set_ylabel("Revenue (₹)")

        ax.bar_label(bars, fmt="₹%.2f", padding=5)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:
        st.warning("price_monitoring_results.csv file not found.")
