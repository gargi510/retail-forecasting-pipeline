"""
Validation Module - Evaluates Ensemble Performance
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from src.utils.metrics import smape, rmse


def run_validation(
    project_root: str,
    xgb_model_path: str,
    nn_model_path: str,
    feature_scaler_path: str,
    target_scaler_path: str,
    nn_features_path: str,
    causal_residual_path: str,
    val_features_file: str,
    val_target_file: str,
    diagnostics_dir: str
):
    """
    Run validation on held-out data
    
    Returns:
        metrics_summary (dict), diagnostics_df (DataFrame)
    """
    
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Load validation data
    X_val = pd.read_parquet(val_features_file)
    y_val = pd.read_parquet(val_target_file)["Sales_Capped"]
    
    # For validation, we need to split the training data
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        X_val, y_val, test_size=0.2, shuffle=False
    )
    
    print(f"Validation samples: {X_val.shape[0]}")
    
    # Load models
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    
    nn_model = tf.keras.models.load_model(nn_model_path)
    
    # Load scalers
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    nn_features = joblib.load(nn_features_path)
    
    # =====================================
    # XGBoost Predictions
    # =====================================
    dval = xgb.DMatrix(X_val)
    xgb_val_preds = xgb_model.predict(dval)
    
    # =====================================
    # Neural Network Predictions
    # =====================================
    X_val_nn = X_val.copy()
    for col in nn_features:
        if col not in X_val_nn.columns:
            X_val_nn[col] = 0
    X_val_nn = X_val_nn[nn_features]
    
    X_val_scaled = feature_scaler.transform(X_val_nn)
    nn_val_scaled = nn_model.predict(X_val_scaled).ravel()
    nn_val_preds = target_scaler.inverse_transform(
        nn_val_scaled.reshape(-1, 1)
    ).ravel()
    
    # =====================================
    # Ensemble
    # =====================================
    ensemble_val_preds = 0.6 * xgb_val_preds + 0.4 * nn_val_preds
    
    # =====================================
    # Compute Metrics
    # =====================================
    metrics_summary = {
        "XGB_RMSE": rmse(y_val, xgb_val_preds),
        "XGB_SMAPE": smape(y_val, xgb_val_preds),
        "NN_RMSE": rmse(y_val, nn_val_preds),
        "NN_SMAPE": smape(y_val, nn_val_preds),
        "ENSEMBLE_RMSE": rmse(y_val, ensemble_val_preds),
        "ENSEMBLE_SMAPE": smape(y_val, ensemble_val_preds)
    }
    
    # =====================================
    # Diagnostics
    # =====================================
    diagnostics_df = pd.DataFrame({
        "y_true": y_val,
        "xgb_pred": xgb_val_preds,
        "nn_pred": nn_val_preds,
        "ensemble_pred": ensemble_val_preds,
        "error": ensemble_val_preds - y_val,
        "abs_error": np.abs(ensemble_val_preds - y_val),
        "pct_error": 100 * (ensemble_val_preds - y_val) / (y_val + 1e-9)
    })
    
    # Model disagreement
    diagnostics_df["model_disagreement"] = np.abs(xgb_val_preds - nn_val_preds)
    
    # Save diagnostics
    diagnostics_path = os.path.join(diagnostics_dir, "ensemble_validation_diagnostics.parquet")
    diagnostics_df.to_parquet(diagnostics_path, index=False)
    
    print(f"✅ Validation diagnostics saved to: {diagnostics_path}")
    
    # Bias check
    print("\n--- BIAS CHECK ---")
    print(diagnostics_df["error"].describe())
    
    # Model stability
    print("\n--- MODEL DISAGREEMENT ---")
    print(diagnostics_df["model_disagreement"].describe())
    
    return metrics_summary, diagnostics_df