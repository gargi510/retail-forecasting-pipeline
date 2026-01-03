# QUICKSTART.md

# Quickstart Guide

Run the Retail Demand Promotion Intelligence System end-to-end in a few minutes.

---

## Prerequisites

Ensure the following are installed:

- **Python 3.10 or higher**
- **Git**
- **Docker** (optional, for containerized runs)

---

## Clone the Repository

```bash
git clone <your-repo-url>
cd Retail-Demand-Promotion-Intelligence-System
```

---

## Local Environment Setup

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac / Linux:**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Setup

Place raw datasets in the following directory:

```
data/raw/
├── TRAIN.csv
└── TEST_FINAL.csv
```

> **Note:** No manual preprocessing is required. All features are generated using leak-safe pipelines.

---

## Run the Full Pipeline

Execute the complete forecasting pipeline with a single command:

```bash
python -m src.pipeline
```

This will automatically:

- Load preprocessed features
- Run XGBoost inference
- Run Neural Network inference
- Apply ensemble weighting
- Generate validation diagnostics
- Save final forecasts

---

## Output Artifacts

### Final Forecasts

```
data/model_output/final_forecasts/
└── final_forecast.parquet
```

### Validation Diagnostics

```
data/model_output/diagnostics/
├── ensemble_validation_diagnostics.parquet
├── bias_summary.txt
└── model_disagreement_stats.txt
```

### Model Artifacts

```
models/artifacts/pretrained_models/
├── xgb_model.json
├── nn_model.keras
├── counterfactual_nn.keras
├── xgb_best_params.pkl
└── top15_feature_importance.png
```

---

## Run Individual Components (Optional)

### Train XGBoost only:

```bash
python -m src.training.train_xgb
```

### Train Neural Network only:

```bash
python -m src.training.train_nn
```

### Run validation only:

```bash
python -m src.validate
```

---


### Build the Docker image:

```bash
docker build -t retail-demand-forecast .
```

### Run the container:

```bash
docker run -v $(pwd)/data:/app/data retail-demand-forecast
```

---

## Expected Performance

| Model              | Validation SMAPE |
|--------------------|------------------|
| XGBoost            | ~5.90%           |
| Neural Network     | ~5.31%           |
| Ensemble (Final)   | ~5.83%           |

---

## Notes

- Feature engineering is fully materialized
- No data leakage during inference
- Ensemble improves stability and bias control
- Production-ready structure with Docker support

---

