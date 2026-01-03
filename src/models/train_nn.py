"""
Neural Network Training Module (Notebook-aligned)
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from src.utils.metrics import smape


def train_nn(project_root: str):
    """
    Train Feedforward Neural Network (matches notebook version for best SMAPE)
    
    Args:
        project_root: Project root directory
    """
    
    # -------------------------
    # Paths
    # -------------------------
    data_dir = os.path.join(project_root, "data/processed")
    model_dir = os.path.join(project_root, "models/artifacts/pretrained_models")
    pred_dir = os.path.join(project_root, "models/artifacts/predictions")
    prep_dir = os.path.join(project_root, "models/artifacts/preprocessors")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)
    
    # -------------------------
    # Load features and target
    # -------------------------
    X_full = pd.read_parquet(os.path.join(data_dir, "features_train_base.parquet"))
    y_full = pd.read_parquet(os.path.join(data_dir, "target_train_base.parquet"))["Sales_Capped"]
    
    # -------------------------
    # Train/Validation split (time-based)
    # -------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.2, shuffle=False
    )
    
    print(f"NN Training samples: {X_tr.shape[0]}, Validation samples: {X_val.shape[0]}")
    
    # -------------------------
    # Scale features & target
    # -------------------------
    feature_scaler = RobustScaler()
    target_scaler = StandardScaler()
    
    X_tr_scaled = feature_scaler.fit_transform(X_tr)
    X_val_scaled = feature_scaler.transform(X_val)
    
    y_tr_scaled = target_scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    
    # Save scalers and feature list
    joblib.dump(feature_scaler, os.path.join(prep_dir, "feature_scaler.save"))
    joblib.dump(target_scaler, os.path.join(prep_dir, "target_scaler.save"))
    joblib.dump(X_full.columns.tolist(), os.path.join(prep_dir, "nn_features_list.save"))
    
    print("✅ Scalers and feature list saved")
    
    # -------------------------
    # Build Neural Network (match notebook)
    # -------------------------
    model = tf.keras.Sequential([
        layers.Input(shape=(X_tr_scaled.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    es = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # -------------------------
    # Train model
    # -------------------------
    print("Training Neural Network...")
    history = model.fit(
        X_tr_scaled, y_tr_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=50,
        batch_size=128,
        callbacks=[es],
        verbose=1
    )
    
    # -------------------------
    # Predictions (inverse transform)
    # -------------------------
    train_preds = target_scaler.inverse_transform(model.predict(X_tr_scaled)).ravel()
    val_preds   = target_scaler.inverse_transform(model.predict(X_val_scaled)).ravel()
    
    # -------------------------
    # Evaluation (SMAPE)
    # -------------------------
    print(f"✅ Train SMAPE: {smape(y_tr, train_preds):.2f}%")
    print(f"✅ Validation SMAPE: {smape(y_val, val_preds):.2f}%")
    
    # -------------------------
    # Save model & predictions
    # -------------------------
    model.save(os.path.join(model_dir, "counterfactual_nn.keras"))
    
    pd.DataFrame({'y_true': y_val, 'nn_pred': val_preds}).to_parquet(
        os.path.join(pred_dir, "val_preds_nn.parquet"), index=False
    )
    
    pd.DataFrame({'y_true': y_tr, 'nn_pred': train_preds}).to_parquet(
        os.path.join(pred_dir, "train_preds_nn.parquet"), index=False
    )
    
    # Save training history
    os.makedirs(os.path.join(model_dir, "training_logs"), exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_parquet(os.path.join(model_dir, "training_logs/nn_training_history.parquet"), index=False)
    
    print("✅ Neural Network model, predictions, and training history saved successfully")
    
    return model
