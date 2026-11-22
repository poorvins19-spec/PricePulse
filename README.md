# PricePulse – Adaptive Dynamic Pricing & Revenue Optimization Engine

## Project Overview

PricePulse is a Machine Learning-based dynamic pricing system designed to recommend optimal product prices based on demand, inventory levels, competitor prices, product categories, and seasonal factors.

Traditional pricing systems use fixed prices and cannot easily respond to changing market conditions. PricePulse addresses this problem by combining machine learning, price optimization, and market trend analysis to support smarter pricing decisions.

## Objectives

- Predict suitable product prices using Machine Learning.
- Analyze the relationship between product price and demand.
- Monitor competitor pricing information.
- Adjust prices based on inventory and demand.
- Detect unusual sales spikes and possible flash sales.
- Explore Reinforcement Learning for dynamic pricing.
- Provide price recommendations through an interactive dashboard.
- Support privacy-conscious pricing through aggregated product data.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas, NumPy | Data processing |
| Scikit-learn | Random Forest model |
| XGBoost | Advanced price prediction |
| TensorFlow | Neural network baseline and DQN experiment |
| Matplotlib | Data visualization |
| FastAPI | Backend API |
| Streamlit | Interactive dashboard |
| SciPy | Price optimization |
| SHAP / Explainable AI | Model interpretation, where configured |

## Project Features

1. **Data Collection:** Uses a CSV dataset containing product pricing, demand, inventory, competitor prices, and sales information.
2. **Data Preprocessing:** Performs data cleaning, encoding, and feature preparation.
3. **Price Prediction:** Uses Random Forest and XGBoost regression models.
4. **Model Comparison:** Evaluates models using MAE, RMSE, and R².
5. **Price Elasticity Analysis:** Examines how changes in price relate to sales quantity.
6. **Competitor Monitoring:** Uses competitor price data to adjust recommendations.
7. **Inventory-Based Pricing:** Applies pricing adjustments based on stock levels.
8. **Flash Sale Detection:** Identifies unusual sales spikes using threshold-based detection.
9. **Price Optimization:** Uses mathematical optimization to find candidate prices.
10. **Reinforcement Learning:** Includes an experimental DQN-based pricing approach.
11. **Market Trend Adaptation:** Adjusts ML predictions using competitor market trends.
12. **Privacy-First Pricing:** Aggregates product information and excludes customer segment information from the privacy-safe output.
13. **API Integration:** Provides prediction services through FastAPI.
14. **Interactive Dashboard:** Displays pricing recommendations and monitoring information through Streamlit.

## Project Structure

```text
PricePulse/
│
├── dataset/
│   ├── pricing_data.csv
│   └── pricing_data_encoded.csv
│
├── app.py
├── api.py
├── eda.py
├── prepare_data.py
├── train_model.py
├── xgboost_model.py
├── compare_models.py
├── price_elasticity.py
├── competitor_monitor.py
├── inventory_adjustment.py
├── flash_sale_detection.py
├── price_monitoring.py
├── demand_sliding_window.py
├── tensorflow_model.py
├── realtime_stream.py
├── flash_sale_stream.py
├── demand_price_adjustment.py
├── reinforcement_learning.py
├── price_optimization.py
├── pricing_explanation.py
├── market_trend_adaptation.py
├── privacy_first.py
│
├── pricepulse_model.pkl
├── pricepulse_xgboost.pkl
├── pricepulse_tensorflow.keras
├── pricepulse_scaler.pkl
├── pricepulse_dqn.keras
├── model_comparison.csv
├── optimized_prices.csv
├── dqn_price_recommendations.csv
├── market_trend_results.csv
├── privacy_safe_pricing_data.csv
│
├── requirements.txt
└── README.md
```

*Note: This is the intended structure. Keep only files that actually exist in your repository.*

## Dataset

The project uses a CSV dataset containing the following attributes:

- Product ID
- Product Category
- Current Price
- Demand
- Inventory
- Competitor Price
- Season
- Customer Segment
- Sales Quantity
- Recommended Price

The dataset supports model training, price prediction, and market analysis.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/PricePulse.git
```

### 2. Open the Project Folder

```bash
cd PricePulse
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Backend

```bash
uvicorn api:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

### 5. Run the Streamlit Dashboard

Open a second terminal:

```bash
streamlit run app.py
```

## Model Evaluation

The project evaluates its machine learning models using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The model comparison results are stored in `model_comparison.csv`.

## Expected Benefits

- Data-driven product pricing recommendations.
- Better understanding of demand and price relationships.
- Improved awareness of competitor pricing.
- Inventory-aware pricing decisions.
- A dashboard for monitoring pricing information.
- A foundation for future real-time pricing systems.

## Current Limitations

This project is a local prototype using CSV-based data and simulated market and streaming scenarios. It does not currently provide a live competitor-price feed, production Apache Flink streaming, or SAP Generative AI Hub integration. The DQN implementation is experimental and should be evaluated before claiming business performance improvements.

## Future Enhancements

- Integrate live competitor pricing APIs.
- Implement Apache Kafka and Apache Flink for real-time processing.
- Improve reinforcement learning with DQN and PPO experiments.
- Integrate SAP Generative AI Hub for pricing explanations.
- Deploy and monitor the system in a production environment.
- Evaluate real-world revenue and stockout improvements.

## Author

**Poorvi Nijaguni Sakkaragi**

B.Tech – Artificial Intelligence and Machine Learning

Sharnbasva University, Kalaburagi

## Project Type

Academic / Machine Learning Project

---

**PricePulse – Smarter Pricing Through Machine Learning.**
