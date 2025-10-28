"""
Validação incremental (walk-forward cumulativa)
Autora: Júlia Valandro Bonzanini
"""

import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ==========================
# CONFIG
# ==========================

class Config:
    DATA_PATH = Path("../../data/processed/air_quality_features_pro.csv")
    MODELS_PATH = Path("../../models/meta_ensemble_pm10_oof_stack.joblib")
    REPORTS_PATH = Path("../../reports/model_refined")
    TARGET = "PM10_Canoas"
    WINTER_MONTHS = (6, 7, 8)

Config.REPORTS_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ==========================
# HELPERS
# ==========================

def compute_metrics(y_true, y_pred):
    bias = float(np.mean(y_pred - y_true))
    return {
        "R2": float(np.clip(r2_score(y_true, y_pred), -5, 1)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": bias
    }

def seasonal_mask(datetimes: pd.Series, winter_months=(6, 7, 8)) -> np.ndarray:
    months = pd.to_datetime(datetimes).dt.month.values
    return np.isin(months, winter_months)

def _ensure_numpy_2d(x):
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x

def _build_meta_features(X_block, dt_block, preds_base, expected_cols):
    parts = [preds_base]
    sin_day = np.sin(2 * np.pi * dt_block.dt.dayofyear.values / 365.0)
    cos_day = np.cos(2 * np.pi * dt_block.dt.dayofyear.values / 365.0)
    parts += [_ensure_numpy_2d(sin_day), _ensure_numpy_2d(cos_day)]

    if "is_winter" in X_block.columns:
        parts.append(_ensure_numpy_2d(X_block["is_winter"].astype(int).values))
    else:
        parts.append(_ensure_numpy_2d(seasonal_mask(dt_block).astype(int)))

    for col in ["PM10_Canoas_roll_mean_3", "PM10_Canoas_roll_mean_7", "frota_index", "frota_centered"]:
        parts.append(_ensure_numpy_2d(X_block.get(col, np.zeros(len(dt_block))).astype(float).values))

    meta = np.hstack(parts)
    if meta.shape[1] < expected_cols:
        pad = np.zeros((meta.shape[0], expected_cols - meta.shape[1]))
        meta = np.hstack([meta, pad])
    elif meta.shape[1] > expected_cols:
        meta = meta[:, :expected_cols]
    return meta


# ==========================
# MAIN
# ==========================

def validate_incremental_stability():
    logging.info("🚀 Iniciando validação incremental (walk-forward cumulativa)...")

    df = pd.read_csv(Config.DATA_PATH, parse_dates=["datetime"])
    df = df.dropna(subset=[Config.TARGET]).copy()

    artifacts = joblib.load(Config.MODELS_PATH)
    scaler = artifacts["scaler"]
    base_models = artifacts["base_models"]
    meta_w = artifacts["meta_winter"]
    meta_nw = artifacts["meta_nonwinter"]
    iso_w = artifacts["iso_winter"]
    iso_nw = artifacts["iso_nonwinter"]

    exp_w = int(getattr(meta_w, "n_features_in_", 8))
    exp_nw = int(getattr(meta_nw, "n_features_in_", 8))
    years = sorted(df["datetime"].dt.year.unique())

    X_df = df.drop(columns=[Config.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[Config.TARGET].values
    dt = pd.to_datetime(df["datetime"])

    incremental_results, drift_index = [], []

    for i in range(1, len(years)):
        train_years = years[:i]
        test_year = years[i]

        df_train = df[df["datetime"].dt.year.isin(train_years)]
        df_test = df[df["datetime"].dt.year == test_year]

        if len(df_train) < 100 or len(df_test) < 20:
            continue

        X_train = df_train.drop(columns=[Config.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_train = df_train[Config.TARGET].values
        dt_train = pd.to_datetime(df_train["datetime"])

        X_test = df_test.drop(columns=[Config.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_test = df_test[Config.TARGET].values
        dt_test = pd.to_datetime(df_test["datetime"])

        # ===== Predição base
        Xs_train = scaler.fit_transform(X_train.values)
        Xs_test = scaler.transform(X_test.values)

        preds_base_train = np.column_stack([
            m.fit(Xs_train, y_train).predict(Xs_train) for _, m in base_models.items()
        ])
        preds_base_test = np.column_stack([
            m.predict(Xs_test) for _, m in base_models.items()
        ])

        meta_train = _build_meta_features(X_train, dt_train, preds_base_train, max(exp_w, exp_nw))
        meta_test = _build_meta_features(X_test, dt_test, preds_base_test, max(exp_w, exp_nw))

        winter_mask_test = seasonal_mask(dt_test, Config.WINTER_MONTHS)
        y_pred = np.zeros_like(y_test, dtype=float)

        if np.any(winter_mask_test):
            y_pred[winter_mask_test] = iso_w.predict(meta_w.predict(meta_test[winter_mask_test]))
        if np.any(~winter_mask_test):
            y_pred[~winter_mask_test] = iso_nw.predict(meta_nw.predict(meta_test[~winter_mask_test]))

        metrics = compute_metrics(y_test, y_pred)
        metrics["Train_Upto"] = f"{min(train_years)}–{max(train_years)}"
        metrics["Test_Year"] = test_year
        incremental_results.append(metrics)

        # ===== Drift adaptativo normalizado
        mu_train = X_train.mean(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
        mu_test = X_test.mean(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)

        relative_diff = (mu_test - mu_train).abs() / (mu_train.abs() + 1e-6)
        drift_val = float(np.clip(relative_diff.mean(), 0, 10))  # escala mais interpretável (0–10)

        drift_index.append({"Train_Upto": metrics["Train_Upto"], "Drift_Index": drift_val})

        logging.info(f"[{metrics['Train_Upto']} → {test_year}] "
                     f"R²={metrics['R2']:.3f} | RMSE={metrics['RMSE']:.2f} | Drift={drift_val:.3f}")

    # ===== Exportação =====
    df_inc = pd.DataFrame(incremental_results)
    df_inc.to_csv(Config.REPORTS_PATH / "incremental_stability.csv", index=False)

    df_drift = pd.DataFrame(drift_index)
    df_drift.to_csv(Config.REPORTS_PATH / "incremental_drift_index.csv", index=False)

    # ===== Gráficos =====
    plt.figure(figsize=(8, 4))
    plt.plot(df_inc["Test_Year"], df_inc["R2"], marker="o", color="teal", label="R²")
    plt.twinx()
    plt.plot(df_inc["Test_Year"], df_inc["RMSE"], marker="s", color="tomato", label="RMSE")
    plt.title("Validação Incremental – Performance Anual")
    plt.xlabel("Ano de Teste")
    plt.ylabel("Métricas (R² / RMSE)")
    plt.tight_layout()
    plt.savefig(Config.REPORTS_PATH / "incremental_stability.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(df_drift["Train_Upto"], df_drift["Drift_Index"], marker="o", color="darkorange")
    plt.title("Evolução do Drift Relativo (Treino→Teste)")
    plt.xlabel("Período de Treino")
    plt.ylabel("Índice Relativo de Drift (0–10)")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(Config.REPORTS_PATH / "incremental_drift.png", dpi=300)
    plt.close()

    logging.info("✅ Resultados incrementais e gráficos exportados com sucesso.")
    return df_inc


if __name__ == "__main__":
    validate_incremental_stability()
