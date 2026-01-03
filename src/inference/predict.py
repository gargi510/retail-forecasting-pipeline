# src/inference/predict.py
import os
import joblib
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from src.features.build_features import build_features

def run_predictions(
    project_root: str,
    xgb_model_path: str,
    nn_model_path: str,
    feature_scaler_path: str,
    target_scaler_path: str,
    nn_features_path: str,
    causal_residual_path: str = None,
    test_features_file: str = None,
    test_raw_file: str = None,
    save_dir: str = None
) -> pd.DataFrame:
    """
    Run ensemble predictions: XGBoost + Neural Network

    Returns:
        forecast_df: DataFrame with predictions
    """

    # -----------------------
    # Load test features
    # -----------------------
    if test_features_file and os.path.exists(test_features_file):
        X_test = pd.read_parquet(test_features_file)
    else:
        # fallback: build features from raw test
        X_test, _, _ = build_features(
            data_dir=os.path.join(project_root, "data/processed"),
            is_train=False
        )

    # -----------------------
    # XGBoost predictions
    # -----------------------
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    xgb_preds = xgb_model.predict(xgb.DMatrix(X_test))

    # -----------------------
    # Neural Network predictions
    # -----------------------
    nn = tf.keras.models.load_model(nn_model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    nn_features = joblib.load(nn_features_path)

    # select features in correct order
    X_test_nn = X_test[nn_features]
    X_test_scaled = feature_scaler.transform(X_test_nn)
    nn_preds = target_scaler.inverse_transform(nn.predict(X_test_scaled)).ravel()

    # -----------------------
    # Ensemble (average)
    # -----------------------
    forecast_df = pd.DataFrame({
        "XGB_Pred": xgb_preds,
        "NN_Pred": nn_preds,
    })
    forecast_df["Ensemble"] = forecast_df.mean(axis=1)

    # -----------------------
    # Save results
    # -----------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        forecast_df.to_parquet(os.path.join(save_dir, "final_forecast.parquet"), index=False)

    return forecast_df
