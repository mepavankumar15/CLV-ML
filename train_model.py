"""
train_model.py — CLV Model Training Pipeline
=============================================
Trains 3 regression models on log-transformed CLV target:
  1. Ridge Regression     (interpretable baseline)
  2. Random Forest        (primary — non-linear, robust)
  3. XGBoost              (gradient boosting — highest accuracy)

Also computes SHAP feature importance and an ensemble prediction.

Run:  python train_model.py  (after generate_data.py)
Output: models/*.pkl, data/customers_clv.csv
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor
import shap

# ── Feature definitions ──────────────────────────────────
FEATURES = [
    'tenure_months',         # How long they've been a customer
    'total_orders',          # Total historical orders
    'total_spend',           # Total historical spend ($)
    'recency_days',          # Days since last purchase
    'age',                   # Customer age
    'income_bracket',        # 1=Low, 2=Mid, 3=Mid-High, 4=High
    'nps_score',             # Net Promoter Score (1-10)
    'online_ratio',          # Proportion of online purchases
    'return_rate',           # Historical return rate
    'support_tickets',       # Number of support interactions
    'discount_usage',        # How often they use discounts
    'avg_order_value',       # Average spend per order ($)
    'tenure_years',          # tenure_months / 12
    'purchase_frequency',    # Orders per year
    'recency_score',         # Quintile score 1-5 (5=most recent)
    'frequency_score',       # Quintile score 1-5 (5=most frequent)
    'monetary_score',        # Quintile score 1-5 (5=highest spend)
    'rfm_score',             # Combined RFM (3-15)
    'retention_rate',        # Estimated retention probability
    'churn_rate',            # 1 - retention_rate
]
TARGET = 'clv_12month'


def evaluate_model(y_true_dollars, y_pred_dollars, model_name):
    """
    Compute and print regression metrics in dollar space.

    Parameters
    ----------
    y_true_dollars : array-like
        Actual CLV values in dollars.
    y_pred_dollars : array-like
        Predicted CLV values in dollars.
    model_name : str
        Display name for the model.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    mae  = mean_absolute_error(y_true_dollars, y_pred_dollars)
    rmse = np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))
    r2   = r2_score(y_true_dollars, y_pred_dollars)
    mape = mean_absolute_percentage_error(y_true_dollars, y_pred_dollars) * 100

    print(f"\n  === {model_name} ===")
    print(f"    MAE:  ${mae:>10,.2f}   (avg dollar error per customer)")
    print(f"    RMSE: ${rmse:>10,.2f}   (penalizes large errors more)")
    print(f"    R²:   {r2:>10.4f}    (proportion of CLV variance explained)")
    print(f"    MAPE: {mape:>10.1f}%   (avg % error per customer)")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def main():
    """Main training pipeline."""

    # ── STEP 1: Load data ────────────────────────────────
    print("=" * 60)
    print("   CLV Model Training Pipeline")
    print("=" * 60)

    try:
        df = pd.read_csv("data/customers.csv")
    except FileNotFoundError:
        print("❌ Error: data/customers.csv not found.")
        print("   Run 'python generate_data.py' first.")
        return

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Target summary:")
    print(f"    Mean:   ${y.mean():,.2f}")
    print(f"    Median: ${y.median():,.2f}")
    print(f"    Min:    ${y.min():,.2f}")
    print(f"    Max:    ${y.max():,.2f}")

    # ── STEP 2: Log-transform the target ─────────────────
    # CLV is right-skewed — log transform makes it more normal
    y_log = np.log1p(y)
    print(f"\n  Original CLV — skew: {y.skew():.3f}")
    print(f"  Log CLV      — skew: {y_log.skew():.3f}")

    # ── STEP 3: Train/test split ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── STEP 4: Scale features ───────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("  Scaler fitted and saved [OK]")

    # Convert test targets back to dollars for evaluation
    y_test_dollars = np.expm1(y_test)

    # ── STEP 5: Train Ridge Regression ───────────────────
    print("\n  Training Ridge Regression (alpha=10.0)...")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    joblib.dump(ridge, "models/ridge_model.pkl")

    y_pred_ridge_log     = ridge.predict(X_test_scaled)
    y_pred_ridge_dollars = np.expm1(y_pred_ridge_log)
    metrics_ridge = evaluate_model(y_test_dollars, y_pred_ridge_dollars, "Ridge Regression")

    # ── STEP 6: Train Random Forest ──────────────────────
    print("\n  Training Random Forest (300 trees)...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, "models/rf_model.pkl")

    y_pred_rf_log     = rf.predict(X_test_scaled)
    y_pred_rf_dollars = np.expm1(y_pred_rf_log)
    metrics_rf = evaluate_model(y_test_dollars, y_pred_rf_dollars, "Random Forest")

    # ── STEP 7: Train XGBoost ────────────────────────────
    print("\n  Training XGBoost (400 trees)...")
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        eval_metric='rmse',
    )
    xgb.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    joblib.dump(xgb, "models/xgb_model.pkl")

    y_pred_xgb_log     = xgb.predict(X_test_scaled)
    y_pred_xgb_dollars = np.expm1(y_pred_xgb_log)
    metrics_xgb = evaluate_model(y_test_dollars, y_pred_xgb_dollars, "XGBoost")

    # ── STEP 8: Ensemble prediction ──────────────────────
    print("\n  Computing Ensemble (50% RF + 30% XGB + 20% Ridge)...")
    y_pred_ensemble_log = (
        0.5 * y_pred_rf_log +
        0.3 * y_pred_xgb_log +
        0.2 * y_pred_ridge_log
    )
    y_pred_ensemble_dollars = np.expm1(y_pred_ensemble_log)
    metrics_ens = evaluate_model(y_test_dollars, y_pred_ensemble_dollars, "Ensemble")

    # ── STEP 9: Cross-validation on RF ───────────────────
    print("\n  Running 5-Fold Cross-Validation on Random Forest...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        rf, X_train_scaled, y_train, cv=kf, scoring='r2'
    )
    print(f"    5-Fold CV R²: {cv_scores.round(4)}")
    print(f"    Mean CV R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── STEP 10: SHAP Feature Importance ─────────────────
    print("\n  Computing SHAP feature importance...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_scaled[:200])

    mean_abs_shap = pd.DataFrame({
        'feature':    FEATURES,
        'importance': np.abs(shap_values).mean(axis=0),
    }).sort_values('importance', ascending=False)

    print("\n  Top 10 CLV drivers (SHAP):")
    print("  " + mean_abs_shap.head(10).to_string(index=False).replace('\n', '\n  '))

    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("  SHAP explainer saved [OK]")

    # ── STEP 11: Save scored dataset ─────────────────────
    print("\n  Scoring all customers with all models...")
    X_all_scaled = scaler.transform(df[FEATURES])

    df['clv_predicted_rf']       = np.expm1(rf.predict(X_all_scaled))
    df['clv_predicted_ridge']    = np.expm1(ridge.predict(X_all_scaled))
    df['clv_predicted_xgb']      = np.expm1(xgb.predict(X_all_scaled))
    df['clv_predicted_ensemble'] = np.expm1(
        0.5 * rf.predict(X_all_scaled) +
        0.3 * xgb.predict(X_all_scaled) +
        0.2 * ridge.predict(X_all_scaled)
    )
    df['clv_error_rf'] = np.abs(df['clv_12month'] - df['clv_predicted_rf'])

    df.to_csv("data/customers_clv.csv", index=False)
    print("  Scored dataset saved to data/customers_clv.csv [OK]")

    # ── STEP 12: Final comparison ────────────────────────
    print("\n" + "=" * 60)
    print("   Model Comparison Summary")
    print("=" * 60)
    print(f"  {'Model':<16} {'MAE ($)':>10} {'RMSE ($)':>11} {'R²':>8} {'MAPE (%)':>10}")
    print("  " + "-" * 56)
    for name, m in [('Ridge', metrics_ridge), ('Random Forest', metrics_rf),
                    ('XGBoost', metrics_xgb), ('Ensemble', metrics_ens)]:
        print(f"  {name:<16} ${m['MAE']:>9,.2f} ${m['RMSE']:>10,.2f} "
              f"{m['R2']:>7.4f} {m['MAPE']:>9.1f}%")
    print("=" * 60)

    # Save metrics for use in app.py
    metrics_all = {
        'Ridge': metrics_ridge,
        'Random Forest': metrics_rf,
        'XGBoost': metrics_xgb,
        'Ensemble': metrics_ens,
    }
    joblib.dump(metrics_all, "models/metrics.pkl")

    print("\n  All models saved [OK]")
    print("  Next: streamlit run app.py")


if __name__ == "__main__":
    main()
