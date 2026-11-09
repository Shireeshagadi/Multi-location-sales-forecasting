# Multi-Location Sales Forecasting Pipeline (Next 3 Months)

## 📌 Project Overview
This project delivers a high-granularity predictive analytics pipeline designed to forecast daily sales for multiple restaurant locations over a 3-month horizon. Built as a capstone project during an **Advanced Professional Training Program (Graduated with Distinction)**, this model captures complex daily fluctuations, weekly seasonality, and localized trends to optimize inventory management and staffing levels.

## 💼 The Business Problem
In the restaurant industry, inaccurate sales forecasting leads to severe operational inefficiencies:
* **Over-forecasting** causes severe food waste, high inventory carrying costs, and over-staffing.
* **Under-forecasting** leads to stockouts of key ingredients, lost revenue, and poor customer satisfaction.

**Objective:** Build a robust, scalable predictive model capable of merging disparate relational datasets to forecast daily sales per item per restaurant, capturing granular fluctuations (weekends, holidays, and temporal patterns).

## 📊 Data Schema & Dimensions
* **Sales Ledger (`sales.csv`):** 109,600 historical transactional records capturing daily operational volumes (`date`, `item_id`, `item_count`, `price`).
* **Menu Catalog (`items.csv`):** 100 individual items mapped to specific restaurant properties with explicit nutritional profiles (`id`, `store_id`, `name`, `kcal`, `cost`).
* **Restaurant Profiles (`resturants.csv`):** 6 active regional commercial storefront locations (`id`, `name`).


## 🛠️ Technical Stack & Methodology
* **Language:** Python
* **Data Wrangling & ETL:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Modeling Approaches:** Linear Regression, Random Forest, XGBoost, and Deep Learning (LSTM)
* **Evaluation Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

### Feature Engineering Highlights
To capture intense daily fluctuations in the data, I engineered several temporal and relational features:
* **Time-Series Features:** Day of week, month, day of year, weekend indicators, and cyclical time encodings.
* **Lag Features:** Historical rolling sales averages to capture short-term operational momentum.
* **Cross-Features:** Interaction metrics blending restaurant location popularity with specific item category trends.

## 📈 Model Performance & Comparison
I systematically evaluated multiple algorithms to find the optimal balance between computational efficiency and predictive accuracy:

| Model Architecture | Primary Metric | Performance Value |
| :--- | :--- | :--- |
| **Linear Regression** | RMSE | 30.84 |
| **Random Forest** | RMSE | 30.75 |
| **XGBoost** | RMSE | 30.75 |
| **LSTM (Deep Learning)** | MAPE | 14.78% |

### Key Insights & Business Outcomes
* **Algorithmic Benchmarking:** Tree-based ensemble models (Random Forest and XGBoost) achieved identical baseline improvements over traditional Linear Regression (RMSE of 30.75 vs 30.84), showing strong handling of non-linear relationship patterns.
* **Sequential Trend Mastery:** The Long Short-Term Memory (LSTM) network successfully modeled the temporal sequencing of daily fluctuations, delivering a highly reliable Mean Absolute Percentage Error (MAPE) of 14.78% over the 3-month forecast window.
* **Operational Impact:** By accurately capturing daily spikes and seasonal demand drops, this forecasting engine allows restaurant operators to make data-driven decisions on inventory ordering schedules, mitigating perishable food waste and maximizing margin efficiency.

## 📁 Repository Structure
```text
├── data/
│   ├── items.csv
│   ├── restaurants.csv
│   └── sales.csv
├── notebooks/
│   └── sales_forecasting_analysis.ipynb  # Data exploration, model benchmarking, and evaluation
├── src/
│   └── forecasting_pipeline.py           # Python script for inference pipeline
└── README.md
```

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
   ```
3. Run the Jupyter Notebook in the `notebooks/` directory to see the step-by-step exploratory data analysis (EDA), model benchmarking, and the final 3-month forecast visualizations.


