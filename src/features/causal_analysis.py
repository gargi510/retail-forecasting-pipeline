"""
Causal Inference Module
Runs fixed-effects regression to compute promotional uplift
with IQR-based clipping on Sales.
"""

import os
import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


def run_causal_analysis(
    train_path: str = "../data/raw/RAW_TRAIN.csv",
    output_dir: str = "../models/artifacts/causal_analysis",
    images_dir: str = "../models/artifacts/images",
    start_date: str = "2019-01-01",
    end_date: str = "2019-05-31"
):
    """
    Runs fixed-effects regression to compute causal promo uplift,
    with Sales IQR clipping, saves residuals, metadata JSON, and plots.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # -----------------------
    # Load raw data
    # -----------------------
    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path, parse_dates=["Date"])

    # -----------------------
    # IQR-based clipping for Sales
    # -----------------------
    if "Sales" not in df.columns:
        raise ValueError("Column 'Sales' not found in train dataset.")

    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df['Sales_Capped'] = df['Sales'].clip(lower=lower_bound, upper=upper_bound)

    # -----------------------
    # Filter analysis window
    # -----------------------
    did_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    print(f"Causal analysis on {len(did_df)} rows from {start_date} to {end_date}")

    # -----------------------
    # Feature engineering
    # -----------------------
    did_df["is_promo"] = (did_df.get("Discount", "No") == "Yes").astype(int)
    did_df["is_friday"] = (did_df["Date"].dt.dayofweek == 4).astype(int)
    did_df["month"] = did_df["Date"].dt.month

    # -----------------------
    # Regression formula
    # -----------------------
    formula = "Sales_Capped ~ is_promo + is_friday + C(Store_Type) + C(Region_Code) + C(month)"

    try:
        model = smf.ols(formula=formula, data=did_df).fit()
    except Exception as e:
        print(f"⚠️ Regression failed: {e}")
        print("Creating dummy causal residuals...")
        did_df["causal_residuals"] = 0
        residual_path = os.path.join(output_dir, "causal_residuals.parquet")
        did_df[["Date", "Store_id", "causal_residuals", "Sales_Capped"]].to_parquet(residual_path, index=False)
        return residual_path, None

    # -----------------------
    # Save residuals
    # -----------------------
    did_df["causal_residuals"] = model.resid
    residual_path = os.path.join(output_dir, "causal_residuals.parquet")
    did_df[["Date", "Store_id", "causal_residuals", "Sales_Capped"]].to_parquet(residual_path, index=False)

    # -----------------------
    # Calculate uplift metrics
    # -----------------------
    promo_coef = model.params.get("is_promo", 0)
    baseline_sales = did_df.loc[did_df["is_promo"] == 0, "Sales_Capped"].mean()
    pct_lift = (promo_coef / baseline_sales * 100) if baseline_sales > 0 else 0

    # Save JSON metadata
    causal_metadata = {
        "incremental_sales_usd": round(float(promo_coef), 2),
        "percentage_lift": round(float(pct_lift), 2),
        "friday_penalty_usd": round(float(model.params.get("is_friday", 0)), 2),
        "model_confidence_p_value": round(float(model.pvalues.get("is_promo", 1.0)), 5),
        "analysis_period": f"{start_date} to {end_date}",
        "methodology": "Fixed-Effects Regression (Causal Inference with IQR clipping)",
        "r_squared": round(float(model.rsquared), 4),
        "observations": int(len(did_df))
    }

    json_path = os.path.join(output_dir, "causal_intelligence.json")
    with open(json_path, "w") as f:
        json.dump(causal_metadata, f, indent=4)

    print(f"✅ Causal uplift: ${promo_coef:.2f} ({pct_lift:.2f}%)")
    print(f"✅ R-squared: {model.rsquared:.4f}")

    # -----------------------
    # Diagnostic plots
    # -----------------------
    # Plot 1: March trends
    if 3 in did_df["month"].values:
        march_trends = did_df[did_df["month"] == 3].groupby(["Date", "is_promo"])["Sales_Capped"].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=march_trends, x="Date", y="Sales_Capped", hue="is_promo", marker="o")
        plt.title("March: Promo vs Non-Promo Sales Trends")
        plt.xlabel("Date")
        plt.ylabel("Average Sales")
        plt.legend(title="Promotion", labels=["No", "Yes"])
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "march_promo_vs_nonpromo.png"), dpi=300)
        plt.close()

    # Plot 2: Residual scatter
    plt.figure(figsize=(10, 4))
    plt.scatter(did_df["Date"], did_df["causal_residuals"], alpha=0.1, s=1)
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Causal Model Residual Analysis")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "causal_residuals.png"), dpi=300)
    plt.close()

    print(f"✅ Causal analysis complete. Files saved:")
    print(f"   - Residuals: {residual_path}")
    print(f"   - Metadata: {json_path}")
    print(f"   - Plots: {images_dir}")

    return residual_path, json_path
