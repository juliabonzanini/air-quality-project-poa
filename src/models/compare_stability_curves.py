"""
Comparação entre estabilidade anual e incremental
Projeto: Modelo Preditivo de Qualidade do Ar – Porto Alegre
Autora: Júlia Valandro Bonzanini
Revisão: Outubro/2025

Funções:
- Combinar métricas de validação anual e incremental
- Gerar gráficos comparativos (R² e RMSE)
- Calcular ganho percentual médio de estabilidade
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# CONFIGURAÇÕES
# ==========================

class Config:
    REPORTS_PATH = Path("../../reports/model_refined")
    OUTPUT_PATH = REPORTS_PATH
    YEARLY_FILE = REPORTS_PATH / "metrics_by_year.csv"
    INCR_FILE = REPORTS_PATH / "incremental_stability.csv"

Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


# ==========================
# FUNÇÕES PRINCIPAIS
# ==========================

def compare_stability_curves():
    logging.info("📈 Iniciando comparação entre estabilidade anual e incremental...")

    df_yearly = pd.read_csv(Config.YEARLY_FILE)
    df_inc = pd.read_csv(Config.INCR_FILE)

    # Padroniza colunas
    df_yearly.rename(columns={"R2": "R2_yearly", "RMSE": "RMSE_yearly"}, inplace=True)
    df_inc.rename(columns={"R2": "R2_incremental", "RMSE": "RMSE_incremental", "Test_Year": "Year"}, inplace=True)

    # Merge coerente
    df_merged = pd.merge(df_yearly, df_inc, on="Year", how="inner")

    # Cálculo de ganho percentual
    df_merged["R2_gain_%"] = 100 * (df_merged["R2_incremental"] - df_merged["R2_yearly"]) / df_merged["R2_yearly"].replace(0, np.nan)
    df_merged["RMSE_reduction_%"] = 100 * (df_merged["RMSE_yearly"] - df_merged["RMSE_incremental"]) / df_merged["RMSE_yearly"].replace(0, np.nan)

    mean_r2_gain = df_merged["R2_gain_%"].mean()
    mean_rmse_reduction = df_merged["RMSE_reduction_%"].mean()

    logging.info(f"📊 Ganho médio de R²: {mean_r2_gain:+.2f}%")
    logging.info(f"📉 Redução média de RMSE: {mean_rmse_reduction:+.2f}%")

    # ==========================
    # GRÁFICOS COMPARATIVOS
    # ==========================

    plt.figure(figsize=(8, 4))
    plt.plot(df_merged["Year"], df_merged["R2_yearly"], marker="o", color="royalblue", label="R² Anual (estático)")
    plt.plot(df_merged["Year"], df_merged["R2_incremental"], marker="s", color="seagreen", label="R² Incremental (cumulativo)")
    plt.title("Comparativo de Estabilidade Temporal (R²)")
    plt.xlabel("Ano")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_PATH / "compare_r2_stability.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df_merged["Year"], df_merged["RMSE_yearly"], marker="o", color="indianred", label="RMSE Anual (estático)")
    plt.plot(df_merged["Year"], df_merged["RMSE_incremental"], marker="s", color="darkorange", label="RMSE Incremental (cumulativo)")
    plt.title("Comparativo de Estabilidade Temporal (RMSE)")
    plt.xlabel("Ano")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_PATH / "compare_rmse_stability.png", dpi=300)
    plt.close()

    # ==========================
    # EXPORTA SÍNTESE
    # ==========================

    df_merged.to_csv(Config.OUTPUT_PATH / "compare_stability_summary.csv", index=False)

    summary = (
        f"\n==== RESUMO COMPARATIVO ====\n"
        f"Ganho médio de R²: {mean_r2_gain:+.2f}%\n"
        f"Redução média de RMSE: {mean_rmse_reduction:+.2f}%\n"
        f"============================\n"
    )
    with open(Config.OUTPUT_PATH / "compare_stability_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    logging.info("✅ Gráficos e resumo comparativo exportados com sucesso.")
    return df_merged


# ==========================
# EXECUÇÃO DIRETA
# ==========================

if __name__ == "__main__":
    compare_stability_curves()
