# 📦 Retail Demand & Inventory Forecasting System

An end-to-end machine learning pipeline and interactive Streamlit command center designed to forecast daily product demand across retail stores, prevent stockout lost sales, and generate automated inventory reorder recommendations.

---

## 🎯 Problem Statement & Business Objective

Retail stores frequently lose revenue when customer demand exceeds on-hand inventory (**stockouts**), or tie up capital when over-ordering slow-moving stock. This project solves both challenges by:
1. **Uncovering demand drivers** through exploratory data analysis across seasonality, promotions, competitor pricing, weather, and day-of-week cycles.
2. **Predicting daily product demand** using a leak-free, time-aware machine learning pipeline.
3. **Translating predictions into operations decisions** via an interactive Streamlit dashboard that calculates stockout gaps, projected revenue, and safety-buffered reorder quantities.

---

## 📊 Project Workflow

### 1. Data Overview & Health Audit (`01_Data_Overview.ipynb`)
* Loaded historical multi-store retail sales data (`sales_data.csv`) covering multiple product categories (`Electronics`, `Clothing`, `Groceries`, `Furniture`, `Toys`) and regions (`North`, `South`, `East`, `West`).
* Audited data types, missing values, duplicate records, and domain constraints (verifying pricing validity and quantifying historical stockout frequency where `Demand > Inventory Level`).

### 2. Exploratory Data Analysis & Time Patterns (`02_EDA_and_Time_Analysis.ipynb`)
* **Univariate & Bivariate Diagnostics:** Analyzed target skewness, category/region balance, competitor price spreads, and demand shifts under active promotions and epidemic alerts.
* **Temporal Decomposition:** Extracted `Year`, `Month`, `Day`, `Day_of_Week`, `Is_Weekend`, and `Quarter` to identify weekly shopping surges and seasonal category shifts.
* **Executive EDA Summary Dashboard:** Built a multi-panel KPI summary tracking Total Demand, Total Revenue, Stockout Lost Sales, and Promotion Lift.

### 3. Feature Engineering & Model Building (`03_Model_Building.ipynb`)
* **Leakage Prevention:**
  * Dropped `Units Sold` (an end-of-day outcome unavailable at forecast time).
  * Used a strict **chronological 80/20 train-test split** so the model trains on past dates and predicts future dates.
  * Computed group-level target encodings strictly on the training fold before mapping onto the test set.
* **Engineered Features:**
  * *Temporal & Cyclical:* `Month_Sin`, `Month_Cos`, `Day_of_Week`, `Is_Weekend`.
  * *Competitive Pricing:* `Effective_Price` (post-discount), `Price_vs_Competitor`, `Price_Ratio`, and `Price_vs_Category`.
  * *Promotion Dynamics:* `Promo_Started` (day-1 promotion spike detector), `Promo_Discount_Power`, and `Promo_Category_Mean`.
  * *Time-Series Memory:* Store-product lag features (`Lag_1`, `Lag_2`, `Lag_7`, `Lag_14`), rolling averages (`7D`, `14D`, `30D`), `Demand_Std_7D` volatility, and `Days_of_Inventory`.
* **Model Progression:**
  * **Baseline (`RandomForestRegressor`):** `R² = 0.6842` | `MAE = 18.52` | `RMSE = 24.79`
  * **Upgraded (`HistGradientBoostingRegressor`):** `R² = 0.8863` | `MAE = 10.84` | `RMSE = 14.90` (with native categorical handling, early stopping, and L2 regularization).
* **Artifact Serialization:** Exported `demand_forecast_model.pkl` bundling the trained model, exact feature column ordering, categorical schemas, lookup maps, and validation metrics.

### 4. Interactive Forecasting Dashboard (`artifact/app.py`)
* Built a lightweight Streamlit web application featuring:
  * **KPI Metric Cards:** Predicted Demand (with `±MAE` confidence band), Projected Revenue, Live Stock Balance (Shortage vs. Surplus), and Recommended Reorder Quantity (with a 15% safety buffer).
  * **Visual Diagnostics:** Demand vs. Inventory momentum chart and Competitive Price positioning chart.
  * **Automated Operations Alerts:** Actionable stockout warnings and pricing sensitivity alerts.

---

## 🗂️ Project Structure

```text
Demand_and_Forecasting/
├── notebooks/
│   ├── 01_Data_Overview.ipynb
│   ├── 02_EDA_and_Time_Analysis.ipynb
│   └── 03_Model_Building.ipynb
├── artifact/
│   ├── app.py
│   └── demand_forecast_model.pkl
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack & Exact Environment

To ensure consistent model serialization between cloud training (Google Colab) and local inference (VS Code), package versions are pinned as follows:

* **Python:** 3.13+
* **NumPy:** `2.5.3`
* **Pandas:** `3.0.6`
* **Scikit-Learn:** `1.9.1`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Web Frontend:** `Streamlit`

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd Demand_and_Forecasting
   ```

2. **Create and activate your environment:**
   ```bash
   conda create -n Data_analysis python=3.13 -y
   conda activate Data_analysis
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit dashboard:**
   ```bash
   python -m streamlit run artifact/app.py
   ```

---

## 🔮 Future Improvements
* **Batch CSV Forecasting:** Allow store managers to upload a multi-store CSV file and export a downloadable reorder sheet.
* **FastAPI + Pydantic Backend:** Decouple model inference into a versioned REST API endpoint (`/predict`, `/health`) for production integration.
* **Cloud Deployment:** Deploy the interactive dashboard on Streamlit Community Cloud / Hugging Face Spaces.
