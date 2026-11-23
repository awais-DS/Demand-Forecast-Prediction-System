from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retail Demand Dashboard", page_icon="📦", layout="wide")

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "demand_forecast_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

artifact = load_model()
model = artifact['model']
feature_cols = artifact['feature_columns']
cat_cols = artifact['categorical_columns']
cat_price_map = artifact['category_Avg_Price_map']
promo_cat_map = artifact['promo_category_map']
cat_dow_map = artifact['category_dow_map']
metrics = artifact['metrics']

st.sidebar.header("🎛️ Forecast Inputs")
store_id = st.sidebar.selectbox("Store ID", ["S001", "S002", "S003", "S004", "S005"])
product_id = st.sidebar.selectbox("Product ID", [f"P{str(i).zfill(4)}" for i in range(1, 21)])
category = st.sidebar.selectbox("Category", list(cat_price_map.keys()))
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])

price = st.sidebar.number_input("Price ($)", 1.0, 500.0, 75.0)
discount = st.sidebar.slider("Discount (%)", 0, 50, 10)
comp_price = st.sidebar.number_input("Competitor Price ($)", 1.0, 500.0, 80.0)
promotion = st.sidebar.checkbox("Promotion Active Today", value=True)
prev_promo = st.sidebar.checkbox("Promotion Active Yesterday", value=False)
epidemic = st.sidebar.checkbox("Epidemic Alert", value=False)

forecast_date = st.sidebar.date_input("Date", pd.to_datetime("2024-01-15"))
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Snowy", "Cloudy"])
season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])

inventory = st.sidebar.number_input("Current Inventory", 0, 2000, 140)
ordered = st.sidebar.number_input("Units Ordered", 0, 2000, 200)
lag_1 = st.sidebar.number_input("Yesterday's Demand", 0, 1000, 120)
rolling_7 = st.sidebar.number_input("7-Day Avg Demand", 0.0, 1000.0, 118.0)

dt = pd.to_datetime(forecast_date)
eff_price = price * (1 - discount / 100.0)
cat_avg = cat_price_map.get(category, eff_price)
promo_int = int(promotion)
dow = dt.dayofweek

row = pd.DataFrame([{
    'Store ID': store_id,
    'Product ID': product_id,
    'Category': category,
    'Region': region,
    'Inventory Level': inventory,
    'Units Ordered': ordered,
    'Price': price,
    'Discount': discount,
    'Weather Condition': weather,
    'Promotion': promo_int,
    'Competitor Pricing': comp_price,
    'Seasonality': season,
    'Epidemic': int(epidemic),
    'Month': dt.month,
    'Day': dt.day,
    'Day_of_Week': dow,
    'Is_Weekend': int(dow >= 5),
    'Month_Sin': np.sin(2 * np.pi * dt.month / 12),
    'Month_Cos': np.cos(2 * np.pi * dt.month / 12),
    'Effective_Price': eff_price,
    'Price_vs_Competitor': eff_price - comp_price,
    'Price_Ratio': eff_price / (comp_price + 1e-5),
    'Promo_Started': int(promotion and not prev_promo),
    'Promo_Discount_Power': promo_int * discount,
    'Demand_Lag_1': lag_1,
    'Demand_Lag_2': lag_1,
    'Demand_Lag_7': rolling_7,
    'Demand_Lag_14': rolling_7,
    'Demand_Rolling_7D': rolling_7,
    'Demand_Rolling_14D': rolling_7,
    'Demand_Rolling_30D': rolling_7,
    'Demand_Std_7D': 12.0,
    'Days_of_Inventory': inventory / (rolling_7 + 1),
    'Category_Avg_Price': cat_avg,
    'Price_vs_Category': eff_price - cat_avg,
    'Promo_Category_Mean': promo_cat_map.get((category, promo_int), rolling_7),
    'Category_DayOfWeek_Mean': cat_dow_map.get((category, dow), rolling_7)
}])

for col in cat_cols:
    row[col] = row[col].astype('category')

pred = max(0, round(float(model.predict(row[feature_cols])[0]), 1))
gap = inventory - pred
reorder = max(0, round((pred * 1.15) - inventory))

st.title("📦 Retail Demand & Inventory Dashboard")
st.caption(f"Model Accuracy: **R² = {metrics['R2']}** | Avg Error: **±{metrics['MAE']} units** | scikit-learn `{artifact['sklearn_version']}`")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Demand", f"{pred:.0f} units", f"±{metrics['MAE']} margin")
c2.metric("Projected Revenue", f"${pred * eff_price:,.2f}", f"${eff_price:.2f} / unit")
c3.metric("Stock Balance", f"{gap:+.0f} units", "Shortage" if gap < 0 else "Surplus", delta_color="normal")
c4.metric("Recommended Order", f"{reorder:.0f} units", "Includes 15% buffer")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Demand vs. Inventory")
    chart_df = pd.DataFrame({
        "Units": [rolling_7, lag_1, pred, inventory]
    }, index=["1. 7-Day Avg", "2. Yesterday", "3. Forecast", "4. Current Stock"])
    st.bar_chart(chart_df)

with right:
    st.subheader("Price Comparison ($)")
    price_df = pd.DataFrame({
        "Price ($)": [price, eff_price, comp_price, cat_avg]
    }, index=["1. List Price", "2. After Discount", "3. Competitor", "4. Category Avg"])
    st.bar_chart(price_df)

if gap < 0:
    st.error(f"🚨 **Stockout Risk:** Order **{reorder:.0f} units** to cover the **{abs(gap):.0f}-unit** shortage.")
else:
    st.success(f"✅ **Stock Healthy:** Current inventory covers demand with **{gap:.0f} units** to spare.")
