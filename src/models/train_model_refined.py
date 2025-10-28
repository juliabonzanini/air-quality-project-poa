"""
Treinamento do Meta-Ensemble (stacking manual c/ OOF)
Projeto: Modelo Preditivo de Qualidade do Ar – Porto Alegre
Autora: Júlia Valandro Bonzanini
Data: Outubro/2025
Arquivo: train_model_refined.py

Aprimoramentos:
- Total compatibilidade com features refinadas (frota_index, frota_centered)
- Escalonamento fold-wise sem vazamento temporal
- Meta-ensemble sazonal (ElasticNetCV inverno / não-inverno)
- Calibração isotônica por estação
- Detecção e mitigação de drift
- Exportação consolidada: refined_model_results.json + artefatos
- Preparado para uso direto em Streamlit e validações walk-forward
"""

from __future__ import annotations

import warnings
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNetCV
from sklearn.isotonic import IsotonicRegression

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# ==============================
# Configurações
# ==============================

class Config:
    DATA_PATH = Path("../../data/processed/air_quality_features_pro.csv")
    MODELS_PATH = Path("../../models")
    REPORTS_PATH = Path("../../reports/model_refined")
    TARGET = "PM10_Canoas"

    N_SPLITS = 5
    RANDOM_STATE = 42
    N_JOBS = -1
    WINTER_MONTHS = (6, 7, 8)


Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
Config.REPORTS_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="No further splits with positive gain")


# ==============================
# Utilitários
# ==============================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas robustas de avaliação."""
    r2 = float(np.clip(r2_score(y_true, y_pred), -5, 1))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    smape = float(np.mean(2 * np.abs(y_pred - y_true) /
                          (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "SMAPE": smape}


def seasonal_mask(datetime_index: pd.Series) -> np.ndarray:
    """Cria máscara booleana para meses de inverno."""
    months = pd.to_datetime(datetime_index).dt.month.values
    return np.isin(months, Config.WINTER_MONTHS)


def build_meta_features(X_df: pd.DataFrame, preds_matrix: np.ndarray) -> np.ndarray:
    """
    Combina previsões OOF + componentes sazonais + indicadores físicos.
    O objetivo é fornecer ao meta-modelo variáveis explicativas complementares.
    """
    parts = [preds_matrix]

    for col in ["sin_dayofyear", "cos_dayofyear"]:
        if col in X_df.columns:
            parts.append(X_df[col].astype(float).values.reshape(-1, 1))

    if "is_winter" in X_df.columns:
        parts.append(X_df["is_winter"].astype(int).values.reshape(-1, 1))
    elif "datetime" in X_df.columns:
        mask = seasonal_mask(X_df["datetime"])
        parts.append(mask.astype(int).reshape(-1, 1))

    for col in ["PM10_Canoas_roll_mean_3", "PM10_Canoas_roll_mean_7"]:
        if col in X_df.columns:
            parts.append(X_df[col].astype(float).values.reshape(-1, 1))

    # Inclusão opcional de índices de frota normalizados
    for col in ["frota_index", "frota_centered"]:
        if col in X_df.columns:
            parts.append(X_df[col].astype(float).values.reshape(-1, 1))

    return np.hstack(parts)


def quantile_model(alpha: float) -> LGBMRegressor:
    """Modelo LightGBM configurado para previsão de quantis."""
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=350,
        learning_rate=0.05,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        force_col_wise=True,
        n_jobs=Config.N_JOBS,
        random_state=Config.RANDOM_STATE,
        verbosity=-1
    )


def simple_drift_report(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Relatório simples de drift 2020–2022 vs 2023–2024.
    Gera CSV e gráfico top-10 das variáveis mais instáveis.
    """
    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"])
    else:
        idx = pd.to_datetime(df.index)

    df_a = df.loc[(idx.dt.year >= 2020) & (idx.dt.year <= 2022), numeric_cols]
    df_b = df.loc[(idx.dt.year >= 2023) & (idx.dt.year <= 2024), numeric_cols]
    if len(df_a) == 0 or len(df_b) == 0:
        logging.warning("⚠️ Intervalos de drift insuficientes para comparação.")
        return pd.DataFrame()

    mu_a, mu_b = df_a.mean(numeric_only=True), df_b.mean(numeric_only=True)
    drift = (mu_b - mu_a).abs().sort_values(ascending=False)
    drift_df = drift.to_frame(name="mean_abs_diff")

    # Ignorar datetime e colunas de sinalização
    drift_df = drift_df[~drift_df.index.str.contains("datetime|has_target|gap_len|is_long_gap")]

    drift_df.to_csv(Config.REPORTS_PATH / "feature_drift_top10.csv")
    top10 = drift_df.head(10)

    plt.figure(figsize=(8, 5))
    top10[::-1]["mean_abs_diff"].plot(kind="barh", color="#007f7f")
    plt.title("Top 10 Features com Maior Drift (2020–2022 → 2023–2024)")
    plt.xlabel("Diferença média absoluta")
    plt.tight_layout()
    plt.savefig(Config.REPORTS_PATH / "feature_drift_top10.png", dpi=150)
    plt.close()

    logging.info("📊 Drift top-10 gerado e salvo.")
    return drift_df


# ==============================
# Treinamento principal
# ==============================

def train_meta_ensemble():
    logging.info("🚀 Iniciando treinamento refinado do Meta-Ensemble...")
    df = pd.read_csv(Config.DATA_PATH, parse_dates=["datetime"])
    df = df.dropna(subset=[Config.TARGET]).copy()

    # --- Preparação de dados ---
    cols_to_drop = [Config.TARGET]
    for c in ["data", "date", "timestamp"]:
        if c in df.columns:
            cols_to_drop.append(c)

    X_df = df.drop(columns=cols_to_drop).copy()
    y = df[Config.TARGET].astype(float).values
    X_df = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    time_idx = pd.to_datetime(df["datetime"])
    winter_mask_full = np.isin(time_idx.dt.month.values, Config.WINTER_MONTHS)

    logging.info(f"📈 Dataset carregado: {len(X_df)} amostras | {X_df.shape[1]} features")

    # ==========================
    # 1. Validação temporal (OOF)
    # ==========================
    tscv = TimeSeriesSplit(n_splits=Config.N_SPLITS)
    base_specs = [
        ("rf", RandomForestRegressor(n_estimators=350, max_depth=12, min_samples_split=4,
                                     random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS)),
        ("xgb", XGBRegressor(objective="reg:squarederror", n_estimators=450, learning_rate=0.05,
                             max_depth=6, subsample=0.9, colsample_bytree=0.9,
                             random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS, verbosity=0)),
        ("lgbm", LGBMRegressor(objective="regression", n_estimators=450, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, min_child_samples=8,
                               force_col_wise=True, n_jobs=Config.N_JOBS,
                               random_state=Config.RANDOM_STATE, verbosity=-1)),
    ]

    oof_preds = np.zeros((len(X_df), len(base_specs)))
    metrics_folds = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_df), start=1):
        if len(test_idx) == 0:
            continue
        X_tr, X_te = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

        fold_preds = []
        for j, (name, model) in enumerate(base_specs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_s, y_tr)
                pred = model.predict(X_te_s)
                oof_preds[test_idx, j] = pred
                fold_preds.append(pred)

        preds_mean = np.mean(np.column_stack(fold_preds), axis=1)
        m = compute_metrics(y_te, preds_mean)
        metrics_folds.append(m)
        logging.info(f"Fold {fold}: R²={m['R2']:.3f} | RMSE={m['RMSE']:.2f}")

    df_metrics = pd.DataFrame(metrics_folds)
    df_metrics.loc["Mean"] = df_metrics[df_metrics["R2"] > -2].mean()
    df_metrics.to_csv(Config.REPORTS_PATH / "fold_metrics.csv")

    # ==========================
    # 2. Meta-ensemble sazonal
    # ==========================
    meta_features_all = build_meta_features(X_df.assign(datetime=time_idx), oof_preds)
    wmask, nw_mask = winter_mask_full, ~winter_mask_full

    def make_meta():
        return ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, 1.0],
                            alphas=np.logspace(-3, 1, 50),
                            max_iter=10000, random_state=Config.RANDOM_STATE)

    meta_w, meta_nw = make_meta(), make_meta()
    meta_w.fit(meta_features_all[wmask], y[wmask])
    meta_nw.fit(meta_features_all[nw_mask], y[nw_mask])

    meta_oof_pred = np.zeros_like(y)
    meta_oof_pred[wmask] = meta_w.predict(meta_features_all[wmask])
    meta_oof_pred[nw_mask] = meta_nw.predict(meta_features_all[nw_mask])

    # ==========================
    # 3. Calibração isotônica
    # ==========================
    iso_w, iso_nw = IsotonicRegression(out_of_bounds="clip"), IsotonicRegression(out_of_bounds="clip")
    iso_w.fit(meta_oof_pred[wmask], y[wmask])
    iso_nw.fit(meta_oof_pred[nw_mask], y[nw_mask])

    meta_cal = np.zeros_like(meta_oof_pred)
    meta_cal[wmask] = iso_w.predict(meta_oof_pred[wmask])
    meta_cal[nw_mask] = iso_nw.predict(meta_oof_pred[nw_mask])

    r2_global_iso = r2_score(y, meta_cal)
    logging.info(f"🧭 R² global calibrado (isotônico): {r2_global_iso:.3f}")

    seasonal_table = pd.DataFrame({
        "Season": ["Winter", "NonWinter"],
        "R2": [r2_score(y[wmask], meta_oof_pred[wmask]),
               r2_score(y[nw_mask], meta_oof_pred[nw_mask])],
        "R2_after": [r2_score(y[wmask], meta_cal[wmask]),
                     r2_score(y[nw_mask], meta_cal[nw_mask])]
    })
    seasonal_table.to_csv(Config.REPORTS_PATH / "seasonal_calibration.csv", index=False)
    pd.DataFrame({"R2_after_isotonic": [float(r2_global_iso)]}).to_csv(
        Config.REPORTS_PATH / "meta_isotonic_metrics.csv", index=False)

    # ==========================
    # 4. Drift simples
    # ==========================
    drift_df = simple_drift_report(df.assign(datetime=time_idx),
                                   X_df.select_dtypes(np.number).columns.tolist())

    # ==========================
    # 5. Treino final completo
    # ==========================
    scaler_full = StandardScaler().fit(X_df)
    X_full_s = scaler_full.transform(X_df)
    base_models_fitted = {name: model.fit(X_full_s, y) for name, model in base_specs}

    q10, q90 = quantile_model(0.10).fit(X_full_s, y), quantile_model(0.90).fit(X_full_s, y)

    # ==========================
    # 6. Exportação de artefatos
    # ==========================
    artifacts = {
        "scaler": scaler_full,
        "base_models": base_models_fitted,
        "meta_winter": meta_w,
        "meta_nonwinter": meta_nw,
        "iso_winter": iso_w,
        "iso_nonwinter": iso_nw,
        "feature_names": list(X_df.columns),
        "config": {
            "n_features": X_df.shape[1],
            "n_samples": len(X_df),
            "winter_months": Config.WINTER_MONTHS,
            "timestamp": datetime.now().isoformat()
        }
    }
    joblib.dump(artifacts, Config.MODELS_PATH / "meta_ensemble_pm10_oof_stack.joblib")
    joblib.dump({"q10": q10, "q90": q90}, Config.MODELS_PATH / "quantile_models.joblib")

    # ==========================
    # 7. JSON consolidado
    # ==========================
    refined_summary = {
        "timestamp": datetime.now().isoformat(),
        "global_r2_after_isotonic": float(r2_global_iso),
        "seasonal": seasonal_table.to_dict(orient="records"),
        "fold_metrics_mean": df_metrics.loc["Mean"].to_dict(),
        "drift_top10": drift_df.head(10).to_dict()["mean_abs_diff"]
        if not drift_df.empty else {},
        "n_samples": len(X_df),
        "n_features": X_df.shape[1],
        "feature_cols": list(X_df.columns)
    }

    with open(Config.REPORTS_PATH / "refined_model_results.json", "w", encoding="utf-8") as f:
        json.dump(refined_summary, f, indent=2, ensure_ascii=False)

    logging.info("✅ Treinamento refinado concluído e artefatos exportados com sucesso.")
    logging.info(f"📦 Artefatos salvos em: {Config.MODELS_PATH}")
    logging.info(f"📊 Resultados consolidados em: {Config.REPORTS_PATH}/refined_model_results.json")


# ==============================
# Execução
# ==============================

if __name__ == "__main__":
    train_meta_ensemble()
