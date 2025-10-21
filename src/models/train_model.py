# src/models/train_model.py
"""
Treinamento de modelos - Baselines e RandomForest (validação temporal)
Autora: Júlia Valandro Bonzanini
Projeto: Modelo Preditivo de Qualidade do Ar (Canoas como proxy para Porto Alegre) - 2020–2024
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from joblib import dump

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# Configurações
# =========================================================

class Config:
    TIMEZONE = "America/Sao_Paulo"

    DATA_FEATURES = Path("../../data/processed/air_quality_features.csv")
    MODELS_DIR = Path("../../models")
    REPORTS_DIR = Path("../../reports")
    FIGURES_DIR = REPORTS_DIR / "figures"
    PROCESSED_DIR = Path("../../data/processed")

    TARGET = "PM10_next_day"
    BASE_COL_PM10 = "PM10_Canoas"

    TEST_SIZE_RATIO = 0.20
    N_SPLITS = 5
    RANDOM_STATE = 42

    DO_AQI_CLASSIF = True
    AQI_BINS = [0, 25, 50, 75, 125, np.inf]
    AQI_LABELS = ["Boa", "Moderada", "Ruim", "Muito Ruim", "Crítica"]

    MODEL_PATH = MODELS_DIR / "random_forest_pm10.joblib"
    METRICS_REG_PATH = REPORTS_DIR / "metrics_regression.json"
    METRICS_CLS_PATH = REPORTS_DIR / "metrics_classification.json"
    PREDICTIONS_PATH = PROCESSED_DIR / "predictions_random_forest.csv"


# =========================================================
# Logging
# =========================================================

def setup_logging():
    Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.REPORTS_DIR / "train_model.log"),
            logging.StreamHandler()
        ],
    )


# =========================================================
# Funções auxiliares
# =========================================================

def regression_metrics(y_true, y_pred) -> dict:
    # Compatível com qualquer versão do sklearn: RMSE = sqrt(MSE)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
    }


def safe_metrics(y_true, y_pred):
    df_tmp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if df_tmp.empty:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}
    return regression_metrics(df_tmp["y_true"], df_tmp["y_pred"])


def to_aqi_category(series: pd.Series) -> pd.Categorical:
    return pd.cut(series, bins=Config.AQI_BINS, labels=Config.AQI_LABELS, include_lowest=True)


# =========================================================
# Visualizações
# =========================================================

def save_regression_plots(df_plot: pd.DataFrame, fig_dir: Path, prefix: str):
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df_plot.index, df_plot["y_true"], label="Observado", alpha=0.8)
    plt.plot(df_plot.index, df_plot["y_pred"], label="Previsto", alpha=0.8)
    plt.title(f"{prefix} - Observado vs. Previsto (Holdout)")
    plt.xlabel("Data")
    plt.ylabel("PM10 (µg/m³)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{prefix}_pred_vs_obs.png", dpi=300)
    plt.close()

    residuals = df_plot["y_true"] - df_plot["y_pred"]
    plt.figure(figsize=(8, 5))
    plt.scatter(df_plot.index, residuals, s=12, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"{prefix} - Resíduos (Holdout)")
    plt.xlabel("Data")
    plt.ylabel("Resíduo (µg/m³)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{prefix}_residuos.png", dpi=300)
    plt.close()


def save_feature_importances(model: RandomForestRegressor, feature_names: list, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    top_feats = [(feature_names[i], importances[i]) for i in order]
    labels = [t[0] for t in top_feats]
    values = [t[1] for t in top_feats]

    plt.figure(figsize=(10, 7))
    plt.barh(labels[::-1], values[::-1])
    plt.title("Importância das Features - Top 20 (RandomForest)")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_feature_importances_top20.png", dpi=300)
    plt.close()


# =========================================================
# Baselines
# =========================================================

def baseline_persistencia(df_test: pd.DataFrame) -> pd.Series:
    if Config.BASE_COL_PM10 not in df_test.columns:
        return pd.Series(index=df_test.index, dtype=float)
    return df_test[Config.BASE_COL_PM10].shift(1)


def baseline_media_movel(df_test: pd.DataFrame, window: int = 3) -> pd.Series:
    if Config.BASE_COL_PM10 not in df_test.columns:
        return pd.Series(index=df_test.index, dtype=float)
    return df_test[Config.BASE_COL_PM10].rolling(window, min_periods=1).mean().shift(1)


# =========================================================
# Execução principal
# =========================================================

def main():
    setup_logging()
    logging.info("Iniciando treinamento...")

    df = pd.read_csv(Config.DATA_FEATURES, parse_dates=["datetime"], index_col="datetime")

    # Garantir timezone consistente (não falhar se não suportado)
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize(Config.TIMEZONE)
        else:
            df.index = df.index.tz_convert(Config.TIMEZONE)
    except Exception:
        pass

    if Config.TARGET not in df.columns:
        raise ValueError(f"Target {Config.TARGET} não encontrado no dataset.")

    # Features numéricas, removendo alvo e quaisquer "next_*"
    candidate_features = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {Config.TARGET}
    future_like = [c for c in candidate_features if "next_" in c]
    drop_cols.update(future_like)
    features = [c for c in candidate_features if c not in drop_cols]

    # Remover linhas sem target
    df = df[df[Config.TARGET].notna()].copy()
    X = df[features].copy()
    y = df[Config.TARGET].copy()

    # Split temporal holdout
    n = len(df)
    test_size = int(np.floor(n * Config.TEST_SIZE_RATIO))
    train_end = n - test_size
    X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

    logging.info(f"Tamanho treino: {len(X_train)}, teste: {len(X_test)}")

    # Baselines
    df_test_full = df.iloc[train_end:]  # inclui colunas base (PM10_Canoas)
    y_pred_persist = baseline_persistencia(df_test_full)
    baseline_persist_metrics = safe_metrics(y_test, y_pred_persist)

    y_pred_mm3 = baseline_media_movel(df_test_full, window=3)
    baseline_mm3_metrics = safe_metrics(y_test, y_pred_mm3)

    y_pred_mm7 = baseline_media_movel(df_test_full, window=7)
    baseline_mm7_metrics = safe_metrics(y_test, y_pred_mm7)

    logging.info(f"Baseline Persistência: {baseline_persist_metrics}")
    logging.info(f"Baseline MM3: {baseline_mm3_metrics}")
    logging.info(f"Baseline MM7: {baseline_mm7_metrics}")

    # RandomForest + GridSearch com TimeSeriesSplit
    pipe = Pipeline(steps=[
        ("var_filter", VarianceThreshold(threshold=0.0)),  # remove colunas com variância zero
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1))
    ])

    param_grid = {
        "rf__n_estimators": [200, 400],
        "rf__max_depth": [8, 12, None],
        "rf__min_samples_leaf": [1, 3, 5]
    }

    tscv = TimeSeriesSplit(n_splits=Config.N_SPLITS)
    gsearch = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",  # usamos MSE e tiramos a raiz depois
        n_jobs=-1,
        verbose=1
    )
    gsearch.fit(X_train, y_train)
    best_model = gsearch.best_estimator_

    best_mse = -gsearch.best_score_
    best_rmse = float(np.sqrt(best_mse))
    logging.info(f"Melhor configuração: {gsearch.best_params_}; CV RMSE: {best_rmse:.3f}")

    # Avaliação holdout
    y_pred_test = best_model.predict(X_test)
    rf_metrics = regression_metrics(y_test, y_pred_test)
    logging.info(f"RandomForest (Holdout): {rf_metrics}")

    # Persistência de artefatos
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_model, Config.MODEL_PATH)

    # Figuras
    plot_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test}, index=X_test.index)
    save_regression_plots(plot_df, Config.FIGURES_DIR, prefix="rf_holdout")

    rf_obj = best_model.named_steps["rf"]
    save_feature_importances(rf_obj, features, Config.FIGURES_DIR)

    # Métricas e previsões
    all_metrics = {
        "random_forest_holdout": rf_metrics,
        "baseline_persistencia": baseline_persist_metrics,
        "baseline_mm3": baseline_mm3_metrics,
        "baseline_mm7": baseline_mm7_metrics,
        "best_params": gsearch.best_params_,
        "cv_rmse": best_rmse
    }
    with open(Config.METRICS_REG_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    out_pred = plot_df.copy()
    out_pred["aqi_pred"] = to_aqi_category(out_pred["y_pred"])
    Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_pred.to_csv(Config.PREDICTIONS_PATH, index=True)

    if Config.DO_AQI_CLASSIF:
        aqi_true = to_aqi_category(plot_df["y_true"])
        aqi_pred = to_aqi_category(plot_df["y_pred"])
        cls_report = classification_report(aqi_true, aqi_pred, labels=Config.AQI_LABELS, output_dict=True, zero_division=0)
        with open(Config.METRICS_CLS_PATH, "w", encoding="utf-8") as f:
            json.dump(cls_report, f, indent=2, ensure_ascii=False)
        logging.info("Relatório de classificação AQI salvo.")

    logging.info("Treinamento concluído com sucesso.")
    logging.info(f"Modelo salvo em: {Config.MODEL_PATH}")


if __name__ == "__main__":
    main()
