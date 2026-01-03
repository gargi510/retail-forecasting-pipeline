# 🛒 Retail Demand Promotion Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosted-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazon-aws)](https://aws.amazon.com/)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-blue?logo=tableau)](https://www.tableau.com/)

An **end-to-end retail sales forecasting & promotion impact intelligence system**, combining **EDA, causal analysis, feature engineering, predictive modeling, and production deployment**.

Designed to help retail teams **maximize promotion effectiveness, optimize inventory, and forecast sales accurately**.

---

## 🚀 Key Highlights

- **Full-cycle pipeline:** Raw data → preprocessing → feature engineering → causal analysis → predictive modeling → automated forecasts
- **Predictive Modeling:** XGBoost + Feedforward Neural Network ensemble achieving **SMAPE 5.83%**
- **Causal Insights:** Measures promotion uplift beyond correlation using causal inference techniques
- **Interactive Dashboards:** Tableau for high-level business insights and executive decision support
- **Deployment-ready:** Dockerized for **AWS Batch**, with optional real-time API capabilities

---

## 📊 Tableau Analysis

**Dashboard:** [Retail Demand Promotion Intelligence Dashboard](https://public.tableau.com/views/RetailDemandPromotionIntelligenceDashboard/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

**Key Insights:**

- 💰 **Promotions drive sales:** ~45% of days are promotional, contributing **52% of total sales**
- 🛒 **Basket size growth:** Average Order Value rises from ~605 → ~649 on discount days
- 🎯 **Holiday trends:** Sales drop **19% on holidays**, even with promotions
- 🌍 **Regional concentration:** Region 1 accounts for **37% of total sales**
- 📈 **Seasonal vs promotional demand:** Promotions peak in March; organic growth peaks in April
- 🔄 **Month-over-Month growth:** Sales +30%, Orders +13%, AOV +15%

> ⚠️ **Note:** Tableau provides **correlations & visual insights**, but not causal analysis or granular forecasting.

---

## 📁 Project Structure

```
Retail-Demand-Promotion-Intelligence-System/
│
├── data/                                    # Data directory
│   ├── raw/                                 # Raw datasets (TRAIN.csv, TEST_FINAL.csv)
│   ├── processed/                           # Preprocessed and feature-engineered data
│   └── model_output/                        # Model predictions and diagnostics
│       ├── final_forecasts/                 # Final forecast outputs
│       └── diagnostics/                     # Validation metrics and reports
│
├── models/                                  # Model artifacts
│   └── artifacts/
│       └── pretrained_models/               # Trained models and feature importance
│           ├── xgb_model.json
│           ├── nn_model.keras
│           ├── counterfactual_nn.keras
│           ├── xgb_best_params.pkl
│           └── top15_feature_importance.png
│
├── Notebooks/                               # Jupyter notebooks for analysis
│   ├── 01_EDA_and_Hypothesis_Testing.ipynb # Exploratory analysis & statistical tests
│   └── 02_causal_analysis.ipynb            # Causal impact analysis
│
├── src/                                     # Source code
│   ├── pipeline.py                          # Main orchestration pipeline
│   ├── preprocessing/                       # Data preprocessing modules
│   ├── features/                            # Feature engineering
│   ├── training/                            # Model training scripts
│   │   ├── train_xgb.py                    # XGBoost training
│   │   └── train_nn.py                     # Neural Network training
│   ├── inference/                           # Model inference
│   ├── ensemble.py                          # Ensemble weighting logic
│   └── validate.py                          # Validation and metrics calculation
│
├── dashboards/                              # Tableau dashboards
│
├── README.md                                # Project overview (this file)
├── QUICKSTART.md                            # Quick setup instructions
├── DEPLOYMENT.md                            # Production deployment guide
├── RESULTS.md                               # Detailed results and performance metrics
├── requirements.txt                         # Python dependencies
├── Dockerfile                               # Docker configuration
└── .gitignore                              # Git ignore rules
```

---

## 🛠️ Components

### 1. **Exploratory Data Analysis (EDA)**

**Notebook:** `Notebooks/01_EDA_and_Hypothesis_Testing.ipynb`

**Objectives:**
- Data validation of 188,340 training records (Jan 2018 – May 2019) and 22,265 test records (June – July 2019)
- Statistical hypothesis testing using ANOVA and Tukey HSD
- Time series analysis including stationarity and autocorrelation tests
- Distribution and correlation analysis

**Key Findings:**
- **Promotional Impact:** Promotions occur on 45% of days but generate 52% of total sales, increasing AOV by ~7%
- **Holiday Performance:** Sales drop 19% on holidays, indicating defensive (not growth-oriented) discounting
- **Regional Trends:** Region 1 contributes 37% of total sales
- **Data Quality:** Clean dataset with no missing or duplicate values

### 2. **Feature Engineering**

**Module:** `src/features/`

Feature types generated:
- **Lag features:** Historical sales patterns
- **Rolling statistics:** Moving averages and standard deviations
- **Categorical features:** Store, item, region encodings
- **Date features:** Day of week, month, seasonality indicators
- **Promotion features:** Discount depth, promotion duration

All features are generated using **leak-safe pipelines** to prevent data leakage.

### 3. **Causal Analysis**

**Notebook:** `Notebooks/02_causal_analysis.ipynb`

- Measures true promotional uplift using causal inference
- Residual analysis to identify incremental impact
- Counterfactual modeling to estimate sales without promotions

### 4. **Predictive Modeling**

#### **XGBoost Model**
- Gradient boosting with hyperparameter tuning (Optuna)
- Validation SMAPE: **~5.90%**
- Script: `src/training/train_xgb.py`

#### **Neural Network Model**
- Feedforward architecture with batch normalization
- RobustScaler for features, StandardScaler for target
- Validation SMAPE: **~5.31%**
- Script: `src/training/train_nn.py`

#### **Ensemble Model**
- Weighted combination of XGBoost + Neural Network
- Improves stability and reduces bias
- Final Validation SMAPE: **~5.83%**
- Script: `src/ensemble.py`

### 5. **Validation & Metrics**

**Script:** `src/validate.py`

Metrics calculated:
- **SMAPE** (Symmetric Mean Absolute Percentage Error)
- **RMSE** (Root Mean Square Error)
- Model disagreement statistics
- Bias summaries

---

## ⚡ Quickstart

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

**TL;DR:**

```bash
# Clone repository
git clone <your-repo-url>
cd Retail-Demand-Promotion-Intelligence-System

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python -m src.pipeline
```

**Outputs:**
- Final forecasts: `data/model_output/final_forecasts/final_forecast.parquet`
- Diagnostics: `data/model_output/diagnostics/`
- Feature importance: `models/artifacts/pretrained_models/top15_feature_importance.png`

---

## 📊 Results

| Model              | Validation SMAPE | Validation RMSE |
|--------------------|------------------|-----------------|
| XGBoost            | ~5.90%           | TBD             |
| Neural Network     | ~5.31%           | TBD             |
| **Ensemble (Final)** | **~5.83%**     | **TBD**         |

> ✅ **SMAPE of 5.83%** indicates highly accurate forecasting suitable for production retail environments.

For detailed performance analysis, see [RESULTS.md](RESULTS.md).

---

## 🏭 Industry Use Cases

### 1. **Inventory Optimization**
Align stock levels with predicted demand to minimize overstock and stockouts, reducing carrying costs and lost sales.

### 2. **Promotion Planning**
Identify high-ROI promotional strategies and optimize campaign timing based on causal uplift analysis.

### 3. **Pricing & Revenue Management**
Balance volume-driven vs. price-driven revenue growth through data-driven pricing decisions.

### 4. **Executive Decision Support**
Tableau dashboards provide actionable insights for strategic planning and resource allocation.

---

## 🐳 Deployment

### Docker Deployment

Build and run the containerized pipeline:

```bash
# Build image
docker build -t retail-demand-forecast .

# Run container
docker run -v $(pwd)/data:/app/data retail-demand-forecast
```

### AWS Batch Deployment

For production-grade deployment on AWS Batch, see [DEPLOYMENT.md](DEPLOYMENT.md).

**Key features:**
- Automated scheduled forecasts
- Scalable batch processing
- Integration with AWS S3 for data storage
- CloudWatch monitoring and logging

### Future Enhancements

- **Real-time API:** Flask/FastAPI endpoint for on-demand predictions
- **Internal Dashboard:** Streamlit app for live KPI monitoring
- **Multi-level Forecasting:** SKU, store, and regional granularity
- **Advanced Models:** LSTM, Temporal Fusion Transformer, causal ML techniques
- **Cloud-native Pipeline:** AWS SageMaker or Lambda for auto-scaling

---

## 📈 Key Features

- ✅ **No data leakage:** Strict train-test separation with leak-safe feature engineering
- ✅ **Causal inference:** Goes beyond correlation to measure true promotional impact
- ✅ **Ensemble modeling:** Combines strengths of gradient boosting and neural networks
- ✅ **Production-ready:** Docker containerization and AWS deployment support
- ✅ **Comprehensive validation:** Multiple metrics and diagnostic reports
- ✅ **Visualization:** Executive dashboards in Tableau

---

## 🔧 Requirements

- Python 3.10 or higher
- Key libraries: XGBoost, Keras (TensorFlow), Scikit-learn, Pandas, NumPy
- Docker (for containerized deployment)
- AWS account (for cloud deployment)

See `requirements.txt` for complete dependency list.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview and introduction (this file) |
| [QUICKSTART.md](QUICKSTART.md) | Fast setup and execution guide |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment instructions |
| [RESULTS.md](RESULTS.md) | Detailed model performance and metrics |

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out via email.

---

## 🙏 Acknowledgments

- XGBoost and TensorFlow communities for excellent ML frameworks
- Tableau for powerful visualization capabilities
- AWS for scalable cloud infrastructure

---

**Built by Gargi Mishra (https://www.linkedin.com/in/gargi510/) for data-driven retail intelligence**