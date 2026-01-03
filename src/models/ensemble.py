# src/models/ensemble.py
import os
from src.inference.predict import run_predictions

def run_ensemble(project_root: str):
    """
    Run ensemble predictions (XGBoost + Neural Network) for production.
    Saves final forecast as parquet.
    """
    # -----------------------
    # Define paths
    # -----------------------
    data_dir = os.path.join(project_root, "data/processed")
    raw_test_path = os.path.join(project_root, "data/raw/TEST_FINAL.csv")
    model_dir = os.path.join(project_root, "models/artifacts")
    output_dir = os.path.join(project_root, "data/model_output/final_forecasts")
    os.makedirs(output_dir, exist_ok=True)

    xgb_model_path = os.path.join(model_dir, "pretrained_models/xgb_model.json")
    nn_model_path = os.path.join(model_dir, "pretrained_models/nn_model.keras")
    feature_scaler_path = os.path.join(model_dir, "preprocessors/feature_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "preprocessors/target_scaler.pkl")
    nn_features_path = os.path.join(model_dir, "preprocessors/nn_features_list.pkl")
    causal_residual_path = os.path.join(model_dir, "causal_analysis/causal_residuals.parquet")
    test_features_file = os.path.join(data_dir, "features_test_base.parquet")

    # -----------------------
    # Run ensemble predictions
    # -----------------------
    forecast_df = run_predictions(
        project_root=project_root,
        xgb_model_path=xgb_model_path,
        nn_model_path=nn_model_path,
        feature_scaler_path=feature_scaler_path,
        target_scaler_path=target_scaler_path,
        nn_features_path=nn_features_path,
        causal_residual_path=causal_residual_path,
        test_features_file=test_features_file,
        test_raw_file=raw_test_path,
        save_dir=output_dir
    )

    print("\n✅ Ensemble Forecast Complete")
    print(f"Forecast saved to: {os.path.join(output_dir, 'final_forecast.parquet')}")
    print("\nSample predictions:")
    print(forecast_df.head(10))

    return forecast_df


if __name__ == "__main__":
    PROJECT_ROOT = "D:/python/Retail Demand Promotion Intelligence System"
    run_ensemble(PROJECT_ROOT)
