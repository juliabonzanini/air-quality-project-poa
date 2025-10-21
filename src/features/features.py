# src/features/features.py
"""
Feature Engineering - Versão Final e Otimizada (com Saúde e Previsão de 3 Dias)
Autora: Júlia Valandro Bonzanini
Projeto: Modelo Preditivo de Qualidade do Ar e Riscos à Saúde em Porto Alegre (2020–2024)
Disciplina: Projeto Integrador de Ciência dos Dados III - UFMS Digital

Descrição:
Gera o conjunto final de features a partir do dataset processado (air_quality_processed.csv),
incluindo variáveis temporais, meteorológicas, de saúde (internações respiratórias),
lags, médias móveis e indicadores externos.
O target é definido como PM10 três dias à frente (PM10_next_3d).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path


# ============================================================
# Configurações globais
# ============================================================

class Config:
    ROOT_PATH = Path(__file__).resolve().parents[2]
    DATA_PATH = ROOT_PATH / "data" / "processed" / "air_quality_processed.csv"
    OUTPUT_PATH = ROOT_PATH / "data" / "processed" / "air_quality_features.csv"
    REPORTS_PATH = ROOT_PATH / "reports"
    TARGET_VAR = "PM10_Canoas"
    USE_CLASSIFICATION = True       # True -> gera AQI_category_next_3d
    DROP_LAST_TARGET_NAN = True     # descarta últimas linhas sem target


# ============================================================
# Logging
# ============================================================

def setup_logging():
    """Configura logging para registrar execução em arquivo e console."""
    Config.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.REPORTS_PATH / "feature_engineering.log"),
            logging.StreamHandler()
        ]
    )


# ============================================================
# Bloco 1 - Features Temporais
# ============================================================

def add_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Gera features temporais (cíclicas e categóricas)."""
    df_temp = pd.DataFrame({
        "year": index.year,
        "month": index.month,
        "day": index.day,
        "weekday": index.weekday,
        "is_weekend": index.weekday.isin([5, 6]).astype(int),
        "dayofyear": index.dayofyear,
        "julian_day": index.to_julian_date().astype(int),
    }, index=index)

    # codificação cíclica
    df_temp["sin_dayofyear"] = np.sin(2 * np.pi * df_temp["dayofyear"] / 365)
    df_temp["cos_dayofyear"] = np.cos(2 * np.pi * df_temp["dayofyear"] / 365)
    return df_temp


# ============================================================
# Bloco 2 - Lags e janelas móveis dos poluentes
# ============================================================

def add_pollutant_features(df: pd.DataFrame, pollutants: list) -> pd.DataFrame:
    """Cria lags, médias e extremos móveis para os poluentes."""
    features = []

    for pol in pollutants:
        if pol not in df.columns:
            continue
        s = df[pol]
        feats = {
            f"{pol}_lag1": s.shift(1),
            f"{pol}_lag2": s.shift(2),
            f"{pol}_lag3": s.shift(3),
            f"{pol}_lag7": s.shift(7),
            f"{pol}_rolling_mean_3": s.rolling(3, min_periods=1).mean(),
            f"{pol}_rolling_mean_7": s.rolling(7, min_periods=1).mean(),
            f"{pol}_rolling_std_7": s.rolling(7, min_periods=1).std(),
            f"{pol}_rolling_min_7": s.rolling(7, min_periods=1).min(),
            f"{pol}_rolling_max_7": s.rolling(7, min_periods=1).max(),
        }
        features.append(pd.DataFrame(feats, index=df.index))

    return pd.concat(features, axis=1) if features else pd.DataFrame(index=df.index)


# ============================================================
# Bloco 3 - Meteorologia e índices compostos
# ============================================================

def add_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera variáveis derivadas de temperatura, umidade, vento e dispersão atmosférica."""
    idx = df.index
    temperatura = df.get("temperatura", pd.Series(np.nan, index=idx))
    umidade = df.get("umidade", pd.Series(np.nan, index=idx))
    vento_vel = df.get("vento_velocidade", pd.Series(np.nan, index=idx))
    vento_dir = df.get("vento_direcao", pd.Series(np.nan, index=idx))
    precip = df.get("precipitacao", pd.Series(np.nan, index=idx))

    temp_diff_daynight = temperatura.rolling(7, min_periods=1).max() - temperatura.rolling(7, min_periods=1).min()
    umidade_inv = 100 - umidade
    vento_log = np.log1p(vento_vel.clip(lower=0))
    dispersao_index = vento_vel * (umidade / 100)
    inversao_proxy = (temp_diff_daynight < 3).astype(float)

    return pd.DataFrame({
        "temperatura": temperatura,
        "umidade": umidade,
        "vento_velocidade": vento_vel,
        "vento_direcao": vento_dir,
        "precipitacao": precip,
        "temp_diff_daynight": temp_diff_daynight,
        "umidade_inv": umidade_inv,
        "vento_log": vento_log,
        "dispersao_index": dispersao_index,
        "inversao_proxy": inversao_proxy,
    }, index=idx)


# ============================================================
# Bloco 4 - Saúde: Internações respiratórias
# ============================================================

def add_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de saúde pública (internações respiratórias)."""
    idx = df.index
    adm = df.get("internacoes_respiratorias", pd.Series(np.nan, index=idx))

    feats = pd.DataFrame({
        "internacoes_respiratorias": adm,
        "internacoes_lag1": adm.shift(1),
        "internacoes_lag3": adm.shift(3),
        "internacoes_roll3": adm.rolling(3, min_periods=1).mean(),
        "internacoes_roll7": adm.rolling(7, min_periods=1).mean(),
    }, index=idx)

    return feats


# ============================================================
# Bloco 5 - Eventos externos (queimadas e feriados)
# ============================================================

def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona focos de queimadas e flag de feriados."""
    idx = df.index
    fire = df.get("focos_queimadas_count", pd.Series(0, index=idx))
    fire_roll7 = fire.rolling(7, min_periods=1).mean()

    return pd.DataFrame({
        "fire_count": fire,
        "fire_count_roll7": fire_roll7,
        "holiday_flag": np.nan,  # preenchido depois
    }, index=idx)


# ============================================================
# Bloco 6 - Target (PM10_next_3d e AQI)
# ============================================================

def add_target(df: pd.DataFrame, target: str, classification: bool = True) -> pd.DataFrame:
    """Cria target de regressão e classificação (AQI) com base no PM10_next_3d."""
    pm10_next_3d = df[target].shift(-3)
    out = {"PM10_next_3d": pm10_next_3d}

    if classification:
        bins = [0, 25, 50, 75, 125, np.inf]
        labels = ["Boa", "Moderada", "Ruim", "Muito Ruim", "Crítica"]
        out["AQI_category_next_3d"] = pd.cut(pm10_next_3d, bins=bins, labels=labels, include_lowest=True)
    else:
        out["AQI_category_next_3d"] = np.nan

    return pd.DataFrame(out, index=df.index)


# ============================================================
# Execução principal do pipeline
# ============================================================

def run_feature_engineering():
    setup_logging()
    logging.info("Iniciando feature engineering final (com saúde e previsão 3 dias)...")

    df_raw = pd.read_csv(Config.DATA_PATH, parse_dates=["datetime"], index_col="datetime")

    # garantir timezone
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize("America/Sao_Paulo")
    else:
        df_raw.index = df_raw.index.tz_convert("America/Sao_Paulo")

    logging.info(f"Dataset carregado: {df_raw.shape[0]} registros, {df_raw.shape[1]} colunas.")

    pollutants = [c for c in df_raw.columns if any(p in c for p in ["PM10", "NO2", "O3", "SO2", "CO"])]

    # gerar blocos
    temporal_blk = add_temporal_features(df_raw.index)
    pollutant_blk = add_pollutant_features(df_raw, pollutants)
    meteo_blk = add_meteorological_features(df_raw)
    health_blk = add_health_features(df_raw)
    external_blk = add_external_features(df_raw)
    target_blk = add_target(df_raw, Config.TARGET_VAR, Config.USE_CLASSIFICATION)

    # concatenar tudo
    df = pd.concat([df_raw, temporal_blk, pollutant_blk, meteo_blk, health_blk, external_blk, target_blk], axis=1)

    # corrigir holiday_flag com base em is_weekend (tratamento seguro)
    if "holiday_flag" in df.columns and "is_weekend" in df.columns:
        is_weekend_series = df["is_weekend"]
        if isinstance(is_weekend_series, pd.DataFrame):
            is_weekend_series = is_weekend_series.iloc[:, 0]
        df["holiday_flag"] = df["holiday_flag"].fillna(is_weekend_series)

    # remover linhas finais sem target
    if Config.DROP_LAST_TARGET_NAN:
        df = df[df["PM10_next_3d"].notna()]

    # defragmentar e salvar
    df = df.copy()
    Config.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Config.OUTPUT_PATH, index=True)
    logging.info(f"Feature set final salvo em {Config.OUTPUT_PATH}")

    return df


# ============================================================
# Execução direta
# ============================================================

if __name__ == "__main__":
    final_df = run_feature_engineering()
    print(final_df.tail())
