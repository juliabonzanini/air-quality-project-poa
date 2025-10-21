"""
Treinamento Refinado de Modelos - Previsão de PM10 (2020–2024)
Autora: Júlia Valandro Bonzanini
Projeto: Modelo Preditivo de Qualidade do Ar e Riscos à Saúde em Porto Alegre
Disciplina: Projeto Integrador III - UFMS Digital

Descrição:
Treina e avalia modelos preditivos multivariados para previsão de PM10 três dias à frente (PM10_next_3d),
utilizando variáveis atmosféricas, meteorológicas, de saúde e eventos externos.
Gera métricas, gráficos e modelos otimizados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

class Config:
    DATA_PATH = Path("../../data/processed/air_quality_features.csv")
    MODEL_PATH = Path("../../models")
    REPORT_PATH = Path("../../reports/model_refined")
    RANDOM_STATE = 42
    TARGET = "PM10_next_3d"

# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    Config.REPORT_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.REPORT_PATH / "train_model_refined.log"),
            logging.StreamHandler()
        ]
    )

# ============================================================
# MÉTRICAS E VISUALIZAÇÕES
# ============================================================

def regression_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred))
    }

def plot_pred_vs_obs(y_true, y_pred, title, save_path):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Observado (µg/m³)")
    plt.ylabel("Previsto (µg/m³)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path / f"{title.replace(' ', '_')}.png", dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, title, save_path):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)[:20]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance.values, y=importance.index, color="steelblue")
        plt.title(title)
        plt.xlabel("Importância Relativa")
        plt.tight_layout()
        plt.savefig(save_path / f"{title.replace(' ', '_')}.png", dpi=300)
        plt.close()

# ============================================================
# PRÉ-PROCESSAMENTO E FEATURE ENGINEERING
# ============================================================

def preprocess_features(df):
    logging.info("Iniciando pré-processamento e limpeza de features...")
    df = df.copy()

    # Remove colunas irrelevantes ou não numéricas
    drop_cols = [c for c in df.columns if "Unnamed" in c or c.lower() in ["estacao", "cidade"]]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # One-hot encoding de variáveis categóricas
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        logging.info(f"Variáveis categóricas codificadas: {list(cat_cols)}")

    # Remove colunas constantes
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index
    if len(const_cols) > 0:
        df.drop(columns=const_cols, inplace=True)
        logging.info(f"Colunas constantes removidas: {list(const_cols)}")

    # Normaliza apenas colunas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    setup_logging()
    logging.info("Iniciando treinamento refinado de modelos com previsão de 3 dias e dados de saúde...")

    df = pd.read_csv(Config.DATA_PATH, parse_dates=["datetime"], index_col="datetime")
    df = df.dropna(subset=[Config.TARGET])

    df = preprocess_features(df)

    X = df.drop(columns=[Config.TARGET])
    y = df[Config.TARGET]

    # Divisão temporal 80/20
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logging.info(f"Tamanho treino: {len(X_train)}, teste: {len(X_test)}")

    # Baselines simples
    y_pred_persist = y_test.shift(1).bfill()
    y_pred_mm3 = y_test.rolling(3, min_periods=1).mean()
    baselines = {
        "Persistência": regression_metrics(y_test, y_pred_persist),
        "Média_Móvel_3d": regression_metrics(y_test, y_pred_mm3)
    }

    # ============================================================
    # TREINAMENTO COM GRIDSEARCH
    # ============================================================

    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "RandomForest": (
            RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [10, 15, None],
                "min_samples_leaf": [2, 4],
            },
        ),
        "XGBoost": (
            XGBRegressor(random_state=Config.RANDOM_STATE, objective="reg:squarederror", n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 8, 12],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.7, 1.0],
            },
        ),
    }

    best_results = {}

    for name, (model, params) in models.items():
        logging.info(f"Treinando modelo: {name}...")
        grid = GridSearchCV(
            model,
            param_grid=params,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            error_score="raise"
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        logging.info(f"{name} melhor configuração: {grid.best_params_}; RMSE(CV): {abs(grid.best_score_):.3f}")

        y_pred = best_model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        logging.info(f"{name} (Holdout): {metrics}")

        best_results[name] = {
            "params": grid.best_params_,
            "metrics": metrics
        }

        # Gráficos e modelos
        plot_pred_vs_obs(y_test, y_pred, f"{name}_Pred_vs_Obs", Config.REPORT_PATH)
        plot_feature_importance(best_model, X.columns, f"{name}_Feature_Importance", Config.REPORT_PATH)
        joblib.dump(best_model, Config.MODEL_PATH / f"{name.lower()}_pm10_next3d.joblib")

    # ============================================================
    # SALVAMENTO DE RESULTADOS
    # ============================================================

    results = {
        "baselines": baselines,
        "models": best_results
    }

    results_path = Config.REPORT_PATH / "refined_model_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logging.info("Treinamento refinado concluído com sucesso.")
    logging.info(f"Resultados e modelos salvos em {Config.REPORT_PATH}")

# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":
    main()
