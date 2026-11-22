


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(
    title="PricePulse API",
    description="Machine Learning Based Dynamic Pricing API",
    version="1.0"
)

# Load trained model
model = joblib.load("pricepulse_model.pkl")


# Input data structure
class ProductInput(BaseModel):
    product_category: str
    current_price: float
    demand: int
    inventory: int
    competitor_price: float
    season: str
    customer_segment: str


# Label mappings used during model training
category_mapping = {
    "Beauty": 0,
    "Electronics": 1,
    "Fashion": 2,
    "Grocery": 3,
    "Home": 4,
    "Sports": 5
}

season_mapping = {
    "Festival": 0,
    "Regular": 1,
    "Summer": 2,
    "Winter": 3
}

segment_mapping = {
    "Budget": 0,
    "Premium": 1,
    "Regular": 2
}


# Home route
@app.get("/")
def home():
    return {
        "message": "Welcome to PricePulse API",
        "status": "Running"
    }


# Price prediction route
@app.post("/predict")
def predict_price(product: ProductInput):

    if product.product_category not in category_mapping:
        return {"error": "Invalid product category"}

    if product.season not in season_mapping:
        return {"error": "Invalid season"}

    if product.customer_segment not in segment_mapping:
        return {"error": "Invalid customer segment"}

    input_data = pd.DataFrame([{
        "product_category": category_mapping[product.product_category],
        "current_price": product.current_price,
        "demand": product.demand,
        "inventory": product.inventory,
        "competitor_price": product.competitor_price,
        "season": season_mapping[product.season],
        "customer_segment": segment_mapping[product.customer_segment]
    }])

    prediction = model.predict(input_data)[0]

    return {
        "project": "PricePulse",
        "recommended_price": round(float(prediction), 2),
        "currency": "INR"
    }
