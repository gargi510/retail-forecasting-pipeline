#!/usr/bin/env python3
"""
Production Pipeline for Retail Demand Forecasting
Orchestrates: Feature Engineering → Training → Validation → Forecasting
"""

import os
from pathlib import Path
import pandas as pd
from src.features.build_features import build_features
from src.models.train_xgb import train_xgb
from src.models.train_nn import train_nn
from src.inference.predict import run_predictions
from src.data.validate import run_validation

def setup_directories(project_root: Path):
    """Create all required directories"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/model_output/final_forecasts",
        "data/model_output/diagnostics",
        "models/artifacts/pretrained_models",
        "models/artifacts/preprocessors",
        "models/artifacts/predictions",
        "models/artifacts/causal_analysis",
        "models/artifacts/images",
        "models/artifacts/meta_features",
        "models/artifacts/training_logs",
    ]
    for d in dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created")

def main():
    PROJECT_ROOT = Path(__file__).parent.absolute()
    setup_directories(PROJECT_ROOT)

    # -----------------------------
    # Paths
    # -----------------------------
    raw_train = PROJECT_ROOT / "data/raw/TRAIN.csv"
    raw_test = PROJECT_ROOT / "data/raw/TEST_FINAL.csv"
    feature_dir = PROJECT_ROOT / "data/processed"

    if not raw_train.exists() or not raw_test.exists():
        raise FileNotFoundError(
            f"Place TRAIN.csv and TEST_FINAL.csv in {PROJECT_ROOT / 'data/raw'}"
        )

    # -----------------------------
    # STEP 1: FEATURE ENGINEERING
    # -----------------------------
    print("\n[STEP 1/6] Building features...")

    df_train = pd.read_csv(raw_train, parse_dates=["Date"])
    df_test = pd.read_csv(raw_test, parse_dates=["Date"])

    # Build features and scale numeric-only columns
    X_train, y_train, train_meta = build_features(df_train, is_train=True)
    X_test, test_meta = build_features(df_test, is_train=False, train_meta=train_meta)

    # Save feature parquet files
    X_train.to_parquet(feature_dir / "features_train_base.parquet", index=False)
    X_test.to_parquet(feature_dir / "features_test_base.parquet", index=False)
    y_train.to_parquet(feature_dir / "target_train_base.parquet", index=False)
    print(f"✅ Features saved: Train {X_train.shape}, Test {X_test.shape}")

    # -----------------------------
    # STEP 2: CAUSAL ANALYSIS
    # -----------------------------
    print("\n[STEP 2/6] Running causal analysis...")
    try:
        from src.features.causal_analysis import run_causal_analysis
        run_causal_analysis(
            train_path=str(raw_train),
            output_dir=str(PROJECT_ROOT / "models/artifacts/causal_analysis"),
            images_dir=str(PROJECT_ROOT / "models/artifacts/images")
        )
    except ImportError:
        print("⚠️ Causal analysis module not found, skipping...")

    # -----------------------------
    # STEP 3: TRAIN XGBOOST
    # -----------------------------
    print("\n[STEP 3/6] Training XGBoost...")
    train_xgb(project_root=str(PROJECT_ROOT))

    # -----------------------------
    # STEP 4: TRAIN NEURAL NETWORK
    # -----------------------------
    print("\n[STEP 4/6] Training Neural Network...")
    train_nn(project_root=str(PROJECT_ROOT))

    # -----------------------------
    # STEP 5: VALIDATION
    # -----------------------------
    print("\n[STEP 5/6] Running validation...")
    metrics, diagnostics = run_validation(
        project_root=str(PROJECT_ROOT),
        xgb_model_path=str(PROJECT_ROOT / "models/artifacts/pretrained_models/xgb_model.json"),
        nn_model_path=str(PROJECT_ROOT / "models/artifacts/pretrained_models/nn_model.keras"),
        feature_scaler_path=str(PROJECT_ROOT / "models/artifacts/preprocessors/feature_scaler.pkl"),
        target_scaler_path=str(PROJECT_ROOT / "models/artifacts/preprocessors/target_scaler.pkl"),
        nn_features_path=str(PROJECT_ROOT / "models/artifacts/preprocessors/nn_features_list.pkl"),
        causal_residual_path=str(PROJECT_ROOT / "models/artifacts/causal_analysis/causal_residuals.parquet"),
        val_features_file=str(feature_dir / "features_train_base.parquet"),
        val_target_file=str(feature_dir / "target_train_base.parquet"),
        diagnostics_dir=str(PROJECT_ROOT / "data/model_output/diagnostics")
    )

    print("\n--- VALIDATION METRICS ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # -----------------------------
    # STEP 6: GENERATE FORECASTS
    # -----------------------------
    print("\n[STEP 6/6] Generating production forecasts...")
    forecast_df = run_predictions(
        project_root=str(PROJECT_ROOT),
        xgb_model_path=str(PROJECT_ROOT / "models/artifacts/pretrained_models/xgb_model.json"),
        nn_model_path=str(PROJECT_ROOT / "models/artifacts/pretrained_models/nn_model.keras"),
        feature_scaler_path=str(PROJECT_ROOT / "models/artifacts/preprocessors/feature_scaler.pkl"),
        target_scaler_path=str(PROJECT_ROOT / "models/artifacts/preprocessors/target_scaler.pkl"),
        nn_features_path=str(PROJECT_ROOT / "models/artifacts/preprocessors/nn_features_list.pkl"),
        causal_residual_path=str(PROJECT_ROOT / "models/artifacts/causal_analysis/causal_residuals.parquet"),
        test_features_file=str(feature_dir / "features_test_base.parquet"),
        test_raw_file=str(raw_test),
        save_dir=str(PROJECT_ROOT / "data/model_output/final_forecasts")
    )

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nForecasts saved to: {PROJECT_ROOT / 'data/model_output/final_forecasts/final_forecast.parquet'}")
    print(f"Diagnostics saved to: {PROJECT_ROOT / 'data/model_output/diagnostics'}")
    print("\nSample forecast:")
    print(forecast_df.head(10))


if __name__ == "__main__":
    main()
