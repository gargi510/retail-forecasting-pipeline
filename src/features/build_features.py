import pandas as pd
import numpy as np
import os

# ===============================================================
# 0. BASE PREPROCESSING + SALES CAPPING
# ===============================================================
def build_features(df: pd.DataFrame, is_train=True, train_meta=None):
    df = df.copy()
    df = df.sort_values(["Store_id", "Date"])

    # -----------------------
    # Cap Sales (Sales_Capped)
    # -----------------------
    if is_train and 'Sales_Capped' not in df.columns:
        Q1 = df['Sales'].quantile(0.25)
        Q3 = df['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['Sales_Capped'] = df['Sales'].clip(lower=lower_bound, upper=upper_bound)
    elif not is_train:
        df['Sales_Capped'] = np.nan  # placeholder for test set

    # -----------------------
    # Categorical typing
    # -----------------------
    cat_cols = ["Store_id", "Store_Type", "Location_Type", "Region_Code"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # -----------------------
    # Discount features
    # -----------------------
    df["Discount_Flag"] = (df.get("Discount") == "Yes").astype("int8") if "Discount" in df.columns else 0
    df["Discount_Holiday"] = df["Discount_Flag"] * df.get("Holiday", 0)
    df["Store_Discount_Flag"] = ((df["Store_Type"] == "S4") & (df["Discount_Flag"] == 1)).astype("int8")

    # -----------------------
    # Calendar features
    # -----------------------
    df["Month"] = df["Date"].dt.month
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    df["Is_Weekend"] = df["DayOfWeek"].isin([5,6]).astype("int8")
    df["Is_Payday"] = df["Date"].dt.day.isin([1,15]).astype("int8")

    # -----------------------
    # Store encodings (TRAIN ONLY)
    # -----------------------
    if is_train:
        # Target encoding
        df["Store_Target_Encoded"] = (
            df.groupby("Store_id")["Sales_Capped"]
              .transform(lambda x: x.shift(1).expanding().mean())
        )
        # Store share of region
        region_mean = df.groupby(["Region_Code","Date"])["Sales_Capped"].transform("mean")
        store_share = df["Sales_Capped"] / (region_mean + 1e-9)
        df["Store_Share_of_Region"] = (
            df.groupby("Store_id")
              .apply(lambda x: store_share.loc[x.index].shift(1).expanding().mean())
              .reset_index(level=0, drop=True)
        )
        # Meta for test set
        stats = df.groupby("Store_id").tail(1)[
            ["Store_id","Store_Target_Encoded","Store_Share_of_Region"]
        ]
        holiday_dates = df.loc[df.get("Holiday",0)==1,"Date"].sort_values().unique()
        meta = {"base_stats": stats, "holiday_dates": holiday_dates}

    else:
        # Merge train stats for test set
        df = df.merge(train_meta["base_stats"], on="Store_id", how="left")
        holiday_dates = train_meta["holiday_dates"]
        meta = None

    # -----------------------
    # Days to next holiday
    # -----------------------
    def days_to_next_holiday(d, holidays):
        future = holidays[holidays >= d]
        return (future[0] - d).days if len(future) > 0 else np.nan

    df["Days_To_Next_Holiday"] = df["Date"].apply(lambda x: days_to_next_holiday(x, holiday_dates))

    # -----------------------
    # Sequence / Lag Features
    # -----------------------
    df = df.sort_values(["Store_id","Date"])
    lag_col = "Sales_Capped"
    grp = df.groupby("Store_id")[lag_col]

    df["Lag_1"]   = grp.shift(1)
    df["Lag_7"]   = grp.shift(7)
    df["Rolling_7"]  = grp.transform(lambda x: x.rolling(7,min_periods=1).mean())
    df["Rolling_30"] = grp.transform(lambda x: x.rolling(30,min_periods=1).mean())
    df["EWMA_7"] = grp.transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["Rolling_Ratio_7_30"] = df["Rolling_7"] / (df["Rolling_30"] + 0.01)
    df["May_Growth_Rate"] = grp.pct_change(30).replace([np.inf,-np.inf],0).clip(-1,5)

    # -----------------------
    # Drop unnecessary columns
    # -----------------------
    drop_cols = [c for c in ["ID","Discount"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # -----------------------
    # Features list
    # -----------------------
    features_base = [
        "Store_id","Store_Type","Location_Type","Region_Code",
        "Month_Sin","Month_Cos","DayOfWeek_Sin","DayOfWeek_Cos",
        "Is_Weekend","Is_Payday",
        "Discount_Flag","Discount_Holiday","Store_Discount_Flag",
        "Store_Target_Encoded","Store_Share_of_Region","Days_To_Next_Holiday"
    ]
    lag_features = [
        "Lag_1","Lag_7","Rolling_7","Rolling_30","EWMA_7",
        "Rolling_Ratio_7_30","May_Growth_Rate"
    ]
    all_features = features_base + lag_features

    # -----------------------
    # Split train/test
    # -----------------------
    if is_train:
        # Remove first 30 rows per store to align lag features
        X_list, y_list = [], []
        for store_id in df["Store_id"].unique():
            store_df = df[df["Store_id"]==store_id].sort_values("Date")
            if len(store_df) > 30:
                X_list.append(store_df.iloc[30:][all_features])
                y_list.append(store_df.iloc[30:]["Sales_Capped"])
        X = pd.concat(X_list)
        y = pd.DataFrame({"Sales_Capped": pd.concat(y_list).values})
    else:
        X = df[all_features]
        y = None

    # -----------------------
    # Encode categoricals
    # -----------------------
    for df_ in [X]:
        for col in df_.select_dtypes(["category"]).columns:
            df_[col] = df_[col].cat.codes

    # -----------------------
    # Final cleaning
    # -----------------------
    f32_max = np.finfo(np.float32).max * 0.7
    X = X.replace([np.inf,-np.inf],np.nan).fillna(0).clip(-f32_max,f32_max)

    return (X, y, meta) if is_train else (X, meta)
