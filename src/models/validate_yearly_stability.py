"""
Validação anual de estabilidade temporal e drift acumulado
Projeto: Modelo Preditivo de Qualidade do Ar – Porto Alegre
Autora: Júlia Valandro Bonzanini
Revisão: Outubro/2025

Avanços:
- Compatível com variáveis frota_index e frota_centered (substitui frota_veicular)
- Preserva coerência com meta-features usadas no treino (sin/cos, inverno, roll_means, frota)
- Gráficos e CSVs sobrescrevem arquivos antigos com segurança
- Relatórios completos para dashboard.py (metrics_by_year, feature_drift_accumulated, gráficos)
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
# CONFIGURAÇÕES
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
# FUNÇÕES AUXILIARES
# ==========================

def compute_metrics(y_true, y_pred):
    return {
        "R2": float(np.clip(r2_score(y_true, y_pred), -5, 1)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }

def seasonal_mask(datetimes: pd.Series) -> np.ndarray:
    months = pd.to_datetime(datetimes).dt.month.values
    return np.isin(months, Config.WINTER_MONTHS)

def _ensure_numpy_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def _pad_or_trim_features(X: np.ndarray, expected_cols: int) -> np.ndarray:
    """Garante compatibilidade dimensional com meta-features do treino."""
    X = np.asarray(X)
    cur = X.shape[1]
    if cur == expected_cols:
        return X
    if cur < expected_cols:
        pad = np.zeros((X.shape[0], expected_cols - cur), dtype=float)
        return np.hstack([X, pad])
    return X[:, :expected_cols]

def _build_meta_features_for_validation(
    X_block: pd.DataFrame,
    datetimes_block: pd.Series,
    preds_base: np.ndarray,
    target_roll3: np.ndarray | None,
    target_roll7: np.ndarray | None,
    expected_cols_w: int,
    expected_cols_nw: int
) -> np.ndarray:
    """
    Replica o build_meta_features do treino v13:
    [preds_base, sin_dayofyear, cos_dayofyear, is_winter,
     PM10_roll_mean_3, PM10_roll_mean_7, frota_index, frota_centered]
    """
    parts = [preds_base]

    sin_day = np.sin(2 * np.pi * datetimes_block.dt.dayofyear.values / 365.0)
    cos_day = np.cos(2 * np.pi * datetimes_block.dt.dayofyear.values / 365.0)
    parts += [_ensure_numpy_2d(sin_day), _ensure_numpy_2d(cos_day)]

    if "is_winter" in X_block.columns:
        winter_flag = X_block["is_winter"].astype(int).values
    else:
        winter_flag = seasonal_mask(datetimes_block).astype(int)
    parts.append(_ensure_numpy_2d(winter_flag))

    # Rolling means (3 e 7)
    r3 = X_block.get("PM10_Canoas_roll_mean_3", pd.Series(np.zeros(len(datetimes_block))))
    r7 = X_block.get("PM10_Canoas_roll_mean_7", pd.Series(np.zeros(len(datetimes_block))))
    parts += [_ensure_numpy_2d(r3.values), _ensure_numpy_2d(r7.values)]

    # Novas features estáveis da frota
    for col in ["frota_index", "frota_centered"]:
        if col in X_block.columns:
            parts.append(_ensure_numpy_2d(X_block[col].astype(float).values))
        else:
            parts.append(_ensure_numpy_2d(np.zeros(len(datetimes_block))))

    meta = np.hstack(parts)
    max_expected = max(expected_cols_w, expected_cols_nw)
    return _pad_or_trim_features(meta, max_expected)

# ==========================
# VALIDAÇÃO ANUAL
# ==========================

def validate_yearly_stability():
    logging.info("Carregando dataset e artefatos do ensemble...")
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

    X_df_all = df.drop(columns=[Config.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_all = df[Config.TARGET].values
    dt_all = pd.to_datetime(df["datetime"])

    target_roll3_all = df[Config.TARGET].rolling(3).mean().values
    target_roll7_all = df[Config.TARGET].rolling(7).mean().values

    years = sorted(df["datetime"].dt.year.unique())
    results = []

    for year in years:
        mask = df["datetime"].dt.year == year
        if mask.sum() < 20:
            continue

        X_block = X_df_all.loc[mask, :]
        y_block = y_all[mask]
        dt_block = dt_all.loc[mask]
        winter_mask = seasonal_mask(dt_block)

        Xs = scaler.transform(X_block.values)
        preds_base = np.column_stack([m.predict(Xs) for _, m in base_models.items()])

        meta_full = _build_meta_features_for_validation(
            X_block=X_block,
            datetimes_block=dt_block,
            preds_base=preds_base,
            target_roll3=target_roll3_all[mask],
            target_roll7=target_roll7_all[mask],
            expected_cols_w=exp_w,
            expected_cols_nw=exp_nw
        )

        Xw = _pad_or_trim_features(meta_full[winter_mask], exp_w)
        Xn = _pad_or_trim_features(meta_full[~winter_mask], exp_nw)

        y_pred = np.zeros_like(y_block, dtype=float)
        if Xw.shape[0] > 0:
            y_pred[winter_mask] = iso_w.predict(meta_w.predict(Xw))
        if Xn.shape[0] > 0:
            y_pred[~winter_mask] = iso_nw.predict(meta_nw.predict(Xn))

        metrics = compute_metrics(y_block, y_pred)
        metrics["Year"] = int(year)
        results.append(metrics)
        logging.info(f"Ano {year}: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.2f}")

    # ==========================
    # MÉTRICAS E GRÁFICO ANUAL
    # ==========================
    df_yearly = pd.DataFrame(results).sort_values("Year")
    df_yearly.to_csv(Config.REPORTS_PATH / "metrics_by_year.csv", index=False)
    logging.info("✅ Métricas anuais exportadas: metrics_by_year.csv")

    plt.figure(figsize=(7, 4))
    plt.plot(df_yearly["Year"], df_yearly["R2"], marker="o", color="teal", label="R²")
    plt.twinx()
    plt.plot(df_yearly["Year"], df_yearly["RMSE"], marker="s", color="tomato", label="RMSE")
    plt.title("Estabilidade Temporal por Ano")
    plt.xlabel("Ano")
    plt.ylabel("Métricas (R² / RMSE)")
    plt.tight_layout()
    plt.savefig(Config.REPORTS_PATH / "yearly_stability.png", dpi=300)
    plt.close()
    logging.info("✅ Gráfico de estabilidade temporal salvo.")

    # ==========================
    # DRIFT ACUMULADO (seguro)
    # ==========================
    num_cols = [
        c for c in X_df_all.select_dtypes(include=[np.number]).columns
        if not c.lower().startswith("date")
    ]
    drift_records = []
    uniq_years = sorted(df["datetime"].dt.year.unique())

    for i in range(1, len(uniq_years)):
        y1, y2 = uniq_years[i - 1], uniq_years[i]
        df_a = X_df_all.loc[df["datetime"].dt.year == y1, num_cols]
        df_b = X_df_all.loc[df["datetime"].dt.year == y2, num_cols]
        if len(df_a) == 0 or len(df_b) == 0:
            continue
        mu_a, mu_b = df_a.mean(), df_b.mean()
        drift = (mu_b - mu_a).abs()
        drift_records.append(drift.rename(f"{y1}->{y2}"))

    if drift_records:
        drift_df = pd.concat(drift_records, axis=1)
        drift_df["mean_abs_diff_accum"] = drift_df.mean(axis=1)
        drift_df.sort_values("mean_abs_diff_accum", ascending=False, inplace=True)
        drift_df.to_csv(Config.REPORTS_PATH / "feature_drift_accumulated.csv")

        # Ignora variáveis artificiais e flags
        drift_df = drift_df[~drift_df.index.str.contains("has_target|gap_len|is_long_gap")]

        top_features = drift_df.head(5)
        logging.info("📊 Top 5 features com maior drift acumulado:")
        for feat, val in top_features["mean_abs_diff_accum"].items():
            logging.info(f"   • {feat}: {val:.4f}")

        drift_path = Config.REPORTS_PATH / "feature_drift_accumulated.png"
        if drift_path.exists():
            drift_path.unlink(missing_ok=True)
            logging.info(f"🧹 Arquivo antigo removido: {drift_path.name}")

        plt.close("all")
        plt.figure(figsize=(8, 5))
        top10 = drift_df.head(10)
        top10["mean_abs_diff_accum"][::-1].plot(kind="barh", color="darkcyan")
        plt.title("Top 10 Features com Maior Drift Acumulado (Ano a Ano)")
        plt.xlabel("Diferença média absoluta acumulada")
        plt.tight_layout()
        plt.savefig(drift_path, dpi=300)
        plt.close("all")
        logging.info(f"✅ Novo gráfico salvo: {drift_path.name}")
    else:
        logging.warning("Não foi possível calcular drift acumulado (anos insuficientes).")

    return df_yearly


if __name__ == "__main__":
    validate_yearly_stability()
