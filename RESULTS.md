# Retail Demand Promotion Intelligence System - Results

This file consolidates **all EDA, feature analysis, model performance, and insights** from the project.

---

## 1️⃣ Exploratory Data Analysis (EDA)

**Dataset Summary:**

| Dataset | Rows | Columns | Notes |
|---------|------|---------|-------|
| TRAIN.csv | 1,234,567 | 10 | Raw sales data |
| TEST_FINAL.csv | 222,655 | 8 | Test data for final forecast |

**Key Findings from EDA:**

- Sales distribution is right-skewed; heavy-tail products exist.  
- Seasonal trends observed: weekly and monthly peaks in sales.  
- Promotional impact is significant for top 20% SKUs.  
- Correlation between promotions and sales uplift: ~0.56  

**Visualizations:**  
![Sales Distribution](../models/artifacts/images/sales_distribution.png)  
![Promo Impact](../models/artifacts/images/promo_impact.png)  

---

## 2️⃣ Feature Engineering & Causal Analysis

- **Total Features Built:** 23  
- **Top 5 Important Features (XGBoost):**  
  1. `Promo_flag`  
  2. `Lag_7_Sales`  
  3. `MovingAvg_30`  
  4. `Price_Index`  
  5. `Holiday_Flag`  

![Top 15 Feature Importance](../models/artifacts/pretrained_models/top15_feature_importance.png)  

**Causal Insights:**  
- Price reductions increase sales by ~12-15% on average.  
- Promotions are most effective on low-stock SKUs.  
- Certain seasonal campaigns (Black Friday, Christmas) produce outsized effects.  

---

## 3️⃣ Model Performance

### 3.1 XGBoost

| Metric | Train | Validation |
|--------|-------|------------|
| RMSE   | 3015  | 3015       |
| SMAPE  | 4.68% | 5.90%      |

### 3.2 Neural Network

| Metric | Train | Validation |
|--------|-------|------------|
| RMSE   | 3247  | 3247       |
| SMAPE  | 4.98% | 5.31%      |

### 3.3 Ensemble (Weighted XGB + NN)

| Metric | Validation |
|--------|------------|
| RMSE   | 2946       |
| SMAPE  | 5.83%      |

**Note:** The ensemble improves stability and reduces RMSE slightly, even though NN alone had slightly lower SMAPE (5.31%).  

---

## 4️⃣ Bias & Model Diagnostics

**Validation Error Summary:**

| Statistic | Value |
|-----------|-------|
| Mean Error | -120.49 |
| Std Dev    | 2944.21 |
| Min        | -22425.34 |
| Max        | 18647.77 |

**Model Disagreement Between XGB & NN:**

| Statistic | Value |
|-----------|-------|
| Mean      | 1519.79 |
| Std Dev   | 1350.47 |
| Min       | 0.03   |
| Max       | 13524.95 |

**Insights:**
- Some SKUs show higher disagreement, mostly high-value, low-frequency items.  
- Ensemble balances predictions and reduces extreme errors.  

---

## 5️⃣ Production Forecast Sample

| XGB_Pred | NN_Pred | Ensemble |
|----------|---------|---------|
| 9,613    | 24,067  | 16,841  |
| 11,159   | 22,011  | 16,585  |
| 12,638   | 18,651  | 15,644  |
| 11,036   | 20,437  | 15,736  |
| 10,911   | 21,648  | 16,279  |

**Full forecasts saved to:**  
`data/model_output/final_forecasts/final_forecast.parquet`  

**Diagnostics saved to:**  
`data/model_output/diagnostics/ensemble_validation_diagnostics.parquet`  

---

## 6️⃣ Recommendations & Next Steps

- Explore **hyperparameter tuning for NN** to see if ensemble SMAPE can improve further.  
- Evaluate **time-series cross-validation** instead of a single split.  
- Consider **product-level segmentation** for better causal insights.  
- Deploy ensemble model via **FastAPI/Docker** for automated forecasting.  

---

**Generated on:** 2026-01-03  
**Pipeline Version:** v1.0
