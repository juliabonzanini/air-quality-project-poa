"""
Pipeline de engenharia de atributos
Autora: Júlia Valandro Bonzanini
Projeto: Modelo Preditivo de Qualidade do Ar - Porto Alegre
Revisão: Outubro/2025

Avanços técnicos:
- Mitigação de drift estrutural da variável 'frota_veicular'
- Criação de 'frota_index' (índice relativo a 2020) e 'frota_centered' (anomalia anual)
- Manutenção da coerência temporal (timezone, integridade diária)
- Preserva a distinção entre dataset full (com lacunas) e pro (para modelagem)
- Compatibilidade total com train_model_refined.py e dashboard.py
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ==============================
# Configurações
# ==============================

class Config:
    TIMEZONE = 'America/Sao_Paulo'
    DATA_PROCESSED_PATH = Path('../../data/processed')
    REPORTS_PATH = Path('../../reports')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ==============================
# Utilitários base
# ==============================

def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.set_index('datetime').sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize(Config.TIMEZONE, nonexistent='shift_forward', ambiguous='NaT')
    else:
        df.index = df.index.tz_convert(Config.TIMEZONE)

    df.index.name = 'datetime'
    return df


def _load_input_csv(input_path: Path) -> pd.DataFrame:
    logging.info(f'📥 Carregando dados de entrada: {input_path}')
    df = pd.read_csv(input_path)
    if 'datetime' not in df.columns:
        raise ValueError("Coluna 'datetime' não encontrada no CSV.")
    return _normalize_datetime_index(df)


def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((b == 0) | (~np.isfinite(b)), np.nan, a / b)


# ==============================
# Blocos de features
# ==============================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['dayofyear'] = df.index.day_of_year
    df['year'] = df.index.year

    df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    def get_season(month):
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:
            return 'spring'

    df['season_label'] = df['month'].apply(get_season)
    df['is_winter'] = (df['season_label'] == 'winter').astype(int)
    df = pd.get_dummies(df, columns=['season_label'], prefix='season', drop_first=False)
    return df


def add_fleet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mitiga drift estrutural da variável frota_veicular, substituindo-a por formas estáveis.
    """
    if 'frota_veicular' not in df.columns:
        logging.warning("⚠️ Coluna 'frota_veicular' ausente. Nenhum ajuste aplicado.")
        return df

    df = _coerce_numeric(df, ['frota_veicular'])
    df['year'] = df.index.year

    base_val = df['frota_veicular'].iloc[0] if df['frota_veicular'].iloc[0] > 0 else df['frota_veicular'].mean()
    if base_val == 0 or np.isnan(base_val):
        base_val = 1.0

    df['frota_index'] = df['frota_veicular'] / base_val
    df['frota_centered'] = df['frota_veicular'] - df.groupby('year')['frota_veicular'].transform('mean')

    df.drop(columns=['frota_veicular'], inplace=True)
    logging.info("🚗 Drift controlado: criada 'frota_index' e 'frota_centered'; removida 'frota_veicular'.")
    return df


def add_lag_rolling_features(df: pd.DataFrame, target_col='PM10_Canoas',
                             lags=(1, 2, 3, 7),
                             roll_windows=(3, 7, 14, 30)) -> pd.DataFrame:
    df = _coerce_numeric(df, [target_col])
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    for w in roll_windows:
        df[f'{target_col}_roll_mean_{w}'] = df[target_col].rolling(w).mean()
        df[f'{target_col}_roll_std_{w}'] = df[target_col].rolling(w).std()
    df['pm10_ratio_3_30'] = safe_div(df.get(f'{target_col}_roll_mean_3'), df.get(f'{target_col}_roll_mean_30'))
    df['pm10_diff_3_30'] = df.get(f'{target_col}_roll_mean_3') - df.get(f'{target_col}_roll_mean_30')
    return df


def add_precipitation_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'precipitacao' not in df.columns:
        return df
    df = _coerce_numeric(df, ['precipitacao'])
    df['precip_roll_sum_7'] = df['precipitacao'].rolling(7).sum()
    df['precip_roll_sum_30'] = df['precipitacao'].rolling(30).sum()
    dry = (df['precipitacao'].fillna(0) == 0).astype(int)
    df['dry_streak'] = dry.groupby((dry != dry.shift()).cumsum()).cumsum()
    df['heavy_rain_event'] = (df['precipitacao'] > df['precipitacao'].rolling(30, min_periods=10).quantile(0.9)).astype(int)
    return df


def add_wind_dispersion_features(df: pd.DataFrame) -> pd.DataFrame:
    if {'vento_velocidade', 'umidade'}.issubset(df.columns):
        df = _coerce_numeric(df, ['vento_velocidade', 'umidade'])
        df['dispersao_index'] = df['vento_velocidade'] * (1 - df['umidade'] / 100)
        df['vento_log'] = np.log1p(df['vento_velocidade'])
        df['dispersao_rel'] = safe_div(df['dispersao_index'], df['vento_log'])
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['temperatura', 'umidade', 'vento_velocidade']
    df = _coerce_numeric(df, [c for c in cols if c in df.columns])
    if {'temperatura', 'PM10_Canoas_lag1'}.issubset(df.columns):
        df['temp_x_pm10lag1'] = df['temperatura'] * df['PM10_Canoas_lag1']
    if {'temperatura', 'umidade'}.issubset(df.columns):
        df['temp_x_umid'] = df['temperatura'] * df['umidade']
    if {'temperatura', 'vento_velocidade'}.issubset(df.columns):
        df['temp_x_vento'] = df['temperatura'] * df['vento_velocidade']
    if 'is_winter' in df.columns and 'umidade' in df.columns:
        df['winter_x_umidade'] = df['is_winter'] * df['umidade']
    if 'is_winter' in df.columns and 'vento_velocidade' in df.columns:
        df['winter_x_vento'] = df['is_winter'] * df['vento_velocidade']
    return df


def add_anomaly_features(df: pd.DataFrame, target_col='PM10_Canoas') -> pd.DataFrame:
    df = _coerce_numeric(df, [target_col])
    month_mean = df.groupby('month')[target_col].transform('mean')
    df['pm10_anomaly'] = df[target_col] - month_mean
    df['pm10_normalized_monthly'] = safe_div(df[target_col], month_mean)

    df['winter_mean_year'] = (
        df[df['is_winter'] == 1]
        .groupby('year')[target_col]
        .transform('mean')
    )
    df['winter_mean_year'] = df['winter_mean_year'].reindex(df.index, method='ffill').fillna(method='bfill')
    df['winter_intensity_index'] = safe_div(df[target_col], df['winter_mean_year'])
    df['seasonal_anomaly'] = df['pm10_anomaly'] * (1 + 0.2 * df['is_winter'])
    return df


def add_correlation_features(df: pd.DataFrame, target_col='PM10_Canoas', roll=7) -> pd.DataFrame:
    def rolling_corr(a, b, window):
        return a.rolling(window).corr(b)
    for col in ['temperatura', 'umidade', 'vento_velocidade']:
        if col in df.columns:
            df = _coerce_numeric(df, [col])
            df[f'corr_{col}_pm10'] = rolling_corr(df[col], df[target_col], roll)
    return df


# ==============================
# Finalização e controle temporal
# ==============================

def finalize_features(df: pd.DataFrame, keep_full_timeline: bool = True) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    target_col = 'PM10_Canoas'
    exog_cols = [c for c in df.columns if c != target_col]

    df[exog_cols] = df[exog_cols].interpolate(method='linear', limit=7, limit_direction='both')
    df[exog_cols] = df[exog_cols].ffill(limit=3).bfill(limit=3)

    df['has_target'] = (~df[target_col].isna()).astype(int)
    nan_run = df[target_col].isna().astype(int)
    run_id = (nan_run != nan_run.shift()).cumsum()
    df['gap_len'] = nan_run.groupby(run_id).transform('sum').where(nan_run == 1, 0)
    df['is_long_gap'] = (df['gap_len'] > 7).astype(int)

    return df if keep_full_timeline else df[df['has_target'] == 1]


def summarize_features(df: pd.DataFrame, output_path: Path):
    summary = pd.DataFrame({
        'feature': df.columns,
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True),
        'non_null': df.notnull().sum()
    })
    summary.to_csv(output_path, index=False)


# ==============================
# Função principal
# ==============================

def generate_features(input_path: Path = None, output_path: Path = None) -> pd.DataFrame:
    if input_path is None:
        input_path = Config.DATA_PROCESSED_PATH / 'air_quality_processed.csv'
    if output_path is None:
        output_path = Config.DATA_PROCESSED_PATH / 'air_quality_features_pro.csv'

    df = _load_input_csv(input_path)

    logging.info('🕒 Adicionando features temporais...')
    df = add_temporal_features(df)

    logging.info('🚗 Corrigindo drift de frota veicular...')
    df = add_fleet_features(df)

    logging.info('📈 Criando lags e médias móveis...')
    df = add_lag_rolling_features(df)

    logging.info('🌧️ Criando features de precipitação...')
    df = add_precipitation_features(df)

    logging.info('💨 Criando índices de dispersão...')
    df = add_wind_dispersion_features(df)

    logging.info('⚙️ Criando interações físico-sazonais...')
    df = add_interactions(df)

    logging.info('📊 Criando anomalias e índices sazonais...')
    df = add_anomaly_features(df)

    logging.info('🔗 Calculando correlações móveis...')
    df = add_correlation_features(df)

    # ===========================
    # Finalização e exportação
    # ===========================
    df_full = finalize_features(df, keep_full_timeline=True)
    df_train = df_full[df_full['has_target'] == 1].copy()

    full_out = Config.DATA_PROCESSED_PATH / 'air_quality_features_full.csv'
    df_full.to_csv(full_out, index=True)
    df_train.to_csv(output_path, index=True)

    summarize_features(df_full, Config.REPORTS_PATH / 'feature_summary.csv')
    missing_by_year = df_full['PM10_Canoas'].isna().astype(int).groupby(df_full.index.year).sum()
    missing_by_year.to_csv(Config.REPORTS_PATH / 'missing_by_year.csv')

    logging.info(f"✅ Salvo: {full_out.name} (full) e {output_path.name} (train)")
    logging.info(f"📏 Linhas (full): {len(df_full)} | Linhas (train): {len(df_train)}")
    logging.info("🏁 Engenharia de atributos finalizada com sucesso.")
    return df_train


# ==============================
# Execução direta
# ==============================

if __name__ == '__main__':
    generate_features()
