"""
Diagnóstico e Análise Avançada do Modelo - RandomForest
Autor: Júlia Valandro Bonzanini
Projeto: Modelo Preditivo de Qualidade do Ar e Riscos à Saúde em Porto Alegre
Data: 2025-10-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import VarianceThreshold

# ===========================================
# CONFIGURAÇÕES
# ===========================================
class Config:
    DATA_PATH = Path("../../data/processed")
    REPORTS_PATH = Path("../../reports/diagnostics")
    MODELS_PATH = Path("../../models")

    PRED_FILE = DATA_PATH / "predictions_random_forest.csv"
    MODEL_FILE = MODELS_PATH / "random_forest_pm10.joblib"

# ===========================================
# LOGGING
# ===========================================
def setup_logging():
    Config.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# ===========================================
# FUNÇÕES DE MÉTRICAS
# ===========================================
def regression_summary(y_true, y_pred):
    """Calcula métricas básicas de regressão."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred))
    }

# ===========================================
# FUNÇÕES DE DIAGNÓSTICO VISUAL
# ===========================================
def residual_distribution(residuals, save_path):
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, kde=True, color="royalblue", bins=25)
    plt.title("Distribuição dos Resíduos - RandomForest")
    plt.xlabel("Resíduo (µg/m³)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(save_path / "residual_distribution.png", dpi=300)
    plt.close()
    logging.info("Distribuição de resíduos salva.")

def residuals_over_time(df, save_path):
    plt.figure(figsize=(10, 5))
    plt.scatter(df["datetime"], df["resid"], s=20, color="dodgerblue", alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Resíduos ao Longo do Tempo (Holdout)", fontsize=12)
    plt.xlabel("Data")
    plt.ylabel("Resíduo (µg/m³)")
    plt.tight_layout()
    plt.savefig(save_path / "residuals_over_time.png", dpi=300)
    plt.close()
    logging.info("Gráfico de resíduos ao longo do tempo salvo.")

def correlation_heatmap(df, save_path):
    """Gera e salva heatmap de correlação numérica."""
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, cbar_kws={'label': 'Correlação'})
    plt.title("Matriz de Correlação - Features Numéricas", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path / "correlation_heatmap.png", dpi=300)
    plt.close()
    logging.info("Heatmap de correlação salvo.")

# ===========================================
# FUNÇÕES DE ANÁLISE DE FEATURE
# ===========================================
def detect_high_correlation(df, threshold=0.95):
    """Detecta pares de features altamente correlacionadas."""
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    correlated_features = [column for column in upper.columns if any(upper[column] > threshold)]
    return correlated_features

def detect_low_variance(df, threshold=0.01):
    """Detecta features com variância baixa."""
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    low_variance_features = df.columns[~selector.get_support()].tolist()
    return low_variance_features

def save_feature_diagnostics(df, save_path):
    """Identifica features redundantes e de baixa variância."""
    logging.info("Analisando colinearidade e variância...")

    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="any")

    high_corr_feats = detect_high_correlation(numeric_df)
    low_var_feats = detect_low_variance(numeric_df)

    summary = {
        "high_correlation_features": high_corr_feats,
        "low_variance_features": low_var_feats,
        "total_high_corr": len(high_corr_feats),
        "total_low_var": len(low_var_feats)
    }

    # Salva relatório
    report_path = save_path / "feature_diagnostics.json"
    pd.Series(summary).to_json(report_path, indent=4)
    logging.info(f"Diagnóstico de features salvo em {report_path}")

    return summary

# ===========================================
# EXECUÇÃO PRINCIPAL
# ===========================================
def main():
    setup_logging()
    logging.info("Iniciando diagnóstico do modelo...")

    # Carrega predições e resíduos
    df = pd.read_csv(Config.PRED_FILE, parse_dates=["datetime"])
    df = df.dropna(subset=["y_true", "y_pred"])
    df["resid"] = df["y_true"] - df["y_pred"]

    # Calcula métricas
    metrics = regression_summary(df["y_true"], df["y_pred"])
    logging.info(f"Métricas de Regressão: {metrics}")

    # Cria diretório
    Config.REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    # Gera gráficos
    residual_distribution(df["resid"], Config.REPORTS_PATH)
    residuals_over_time(df, Config.REPORTS_PATH)
    correlation_heatmap(df.select_dtypes(include=[np.number]), Config.REPORTS_PATH)

    # Diagnóstico de features
    feature_summary = save_feature_diagnostics(df, Config.REPORTS_PATH)

    # Salva métricas gerais
    metrics_path = Config.REPORTS_PATH / "diagnostic_metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=4)
    logging.info(f"Métricas salvas em {metrics_path}")

    # Resumo final
    logging.info("Diagnóstico concluído.")
    logging.info(f"Features com alta correlação: {feature_summary['total_high_corr']}")
    logging.info(f"Features com baixa variância: {feature_summary['total_low_var']}")

# ===========================================
# RUN
# ===========================================
if __name__ == "__main__":
    main()
