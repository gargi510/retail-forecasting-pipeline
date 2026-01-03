import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import optuna
from src.utils.metrics import smape  # your existing SMAPE function

def train_xgb(project_root: str, n_trials: int = 25):
    """
    Train XGBoost on preprocessed features with Optuna hyperparameter tuning to minimize SMAPE.
    """

    data_dir  = os.path.join(project_root, "data/processed")
    model_dir = os.path.join(project_root, "models/artifacts/pretrained_models")
    pred_dir  = os.path.join(project_root, "models/artifacts/predictions")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # -----------------------
    # Load feature parquet files
    # -----------------------
    X = pd.read_parquet(os.path.join(data_dir, "features_train_base.parquet"))
    y = pd.read_parquet(os.path.join(data_dir, "target_train_base.parquet"))["Sales_Capped"]

    # -----------------------
    # Train/Validation split
    # -----------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # -----------------------
    # Optuna objective function
    # -----------------------
    def objective(trial):
        params = {
            "n_estimators": 2000,
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 15),
            "gamma": trial.suggest_float("gamma", 0.5, 3.0),
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 42
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        val_pred = model.predict(X_val)
        return smape(y_val, val_pred)

    # -----------------------
    # Run Optuna tuning
    # -----------------------
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("✅ Best Optuna trial parameters:")
    print(study.best_trial.params)

    # -----------------------
    # Train final model with best params
    # -----------------------
    best_params = study.best_trial.params
    best_params.update({
        "n_estimators": 2000,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42
    })

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

    # -----------------------
    # Evaluate
    # -----------------------
    train_preds = final_model.predict(X_tr)
    val_preds   = final_model.predict(X_val)
    print(f"TRAIN SMAPE: {smape(y_tr, train_preds):.2f}%")
    print(f"VAL   SMAPE: {smape(y_val, val_preds):.2f}%")

    # -----------------------
    #  Save top 15 feature importances
    # -----------------------
    feature_importances = final_model.feature_importances_
    feature_names = X_tr.columns

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(10,6))
    plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="skyblue")
    plt.xlabel("Importance")
    plt.title("Top 15 XGBoost Feature Importances")
    plt.tight_layout()

    plt_path = os.path.join(model_dir, "top15_feature_importance.png")
    plt.savefig(plt_path)
    plt.close()
    print(f"✅ Top 15 feature importance saved to {plt_path}")

    # -----------------------
    # Save model & validation preds
    # -----------------------
    final_model.get_booster().save_model(os.path.join(model_dir, "xgb_model.json"))
    pd.DataFrame({
        "y_true": y_val,
        "xgb_pred": val_preds
    }).to_parquet(os.path.join(pred_dir, "val_preds_xgb.parquet"), index=False)

    print("✅ XGBoost training with Optuna tuning complete!")

    return final_model, study.best_trial.params
